"""Stage the accepted 039 release privately, then publish and verify its pinned commit."""
import argparse
from base64 import b64encode
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import atomic_json, single_writer
from one_gpu_lm.models.modelscope_readback import file_path, model_endpoint, readback, validate_files

from release_package import BASE_LICENSE, ENDPOINT, FILES, OWNER, read, require


BRANCH = 'master'
PRIVATE, PUBLIC = 1, 5
INITIAL_FILES = {'.gitattributes','README.md','configuration.json'}
RELEASE_GATES = {'fresh_pairs','historical_pairs','fresh_gain','fresh_regressions','historical_regressions',
                 'format_failure_rate','legacy','save_reload_greedy_match','rights_and_secret_scan_pass',
                 'all_execution_audits_pass','single_training_seed_and_repository_cluster_limits_disclosed'}


def verify_tree(directory, files):
    require(directory.is_dir() and not directory.is_symlink(),'Expected a real release directory')
    entries = list(directory.rglob('*'))
    require(not any(p.is_symlink() or not (p.is_dir() or p.is_file()) for p in entries),
            'Release files must be regular files without symlinks')
    actual = {p.relative_to(directory).as_posix():p for p in entries if p.is_file()}
    require(actual.keys() == files.keys(),'The local payload inventory differs from the release receipt')
    for name,entry in files.items():
        path = actual[name]
        require(path.stat().st_size == entry['bytes'] and sha256(path) == entry['sha256'],
                'The local release bytes differ from the receipt: '+name)


def approved_package(directory):
    evidence_path = directory/'release-evidence.json'
    evidence = read(evidence_path)
    manifest,decision = evidence['manifest'],evidence['decision']
    repo_id = manifest['repo_id']
    model_endpoint(ENDPOINT,repo_id)
    require(evidence['schema_version'] == 1 and evidence['final_test_opened'] is True
            and evidence['weights_public'] is False and decision['accepted'] is True
            and decision['failed_gates'] == [] and set(decision['gates']) == RELEASE_GATES
            and all(value is True for value in decision['gates'].values())
            and manifest['endpoint'] == ENDPOINT and repo_id.split('/')[0] == OWNER,
            'Require an accepted 039 release for the authorized ModelScope account')
    files = manifest['files']
    validate_files(files)
    required = {'README.md',BASE_LICENSE,*FILES}
    require(required <= files.keys() and all(name.startswith('training-source-notices/')
                                            for name in files.keys()-required),
            'The expected payload is the adapter, model card and reviewed licenses')
    adapter = evidence['adapter']
    require({name:files[name]['sha256'] for name in FILES} == decision['checkpoint_sha256']
            == adapter['adapter_files_sha256'] and adapter['model_id'] == decision['model_id']
            and adapter['revision'] == decision['base_revision'],
            'Expected the two accepted adapter files')
    captures = evidence['reload_captures_sha256']
    require(len(captures) == 2 and len(set(captures)) == 2 and all(re.fullmatch(r'[0-9a-f]{64}',digest)
            for digest in (evidence['execution_audit_sha256'],evidence['rights_review_sha256'],*captures)),
            'Require the final audit, rights review and distinct reload receipts')
    license_id = evidence['release_license'].lower()
    require(re.fullmatch(r'[a-z0-9][a-z0-9.+-]*',license_id),'Expected the approved release license')
    verify_tree(directory/'payload',files)
    return manifest,license_id,sha256(evidence_path)


def repository(hub, repo_id):
    from modelscope_hub.errors import NotExistError

    try:
        return hub.get_repo(repo_id,'model')
    except NotExistError:
        return None


def publishing_hub():
    from modelscope_hub import HubApi

    class PrivateUploadHub(HubApi):
        def _create_repo_exist_ok(self, repo_id, repo_type):
            # The pinned upload SDK invokes this hook before sending files.
            info = repository(self, repo_id) if repo_type == 'model' else None
            require(info is not None and info.visibility == PRIVATE,
                    'Upload requires the existing private ModelScope model repository')

    return PrivateUploadHub(endpoint=ENDPOINT)


def branch_head(repo_id, token=None, *, allow_empty=False):
    model_endpoint(ENDPOINT,repo_id)
    url = f'{ENDPOINT}/models/{repo_id}.git'
    env = {k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_CONFIG_GLOBAL=os.devnull,GIT_CONFIG_NOSYSTEM='1',GIT_TERMINAL_PROMPT='0',GIT_CONFIG_COUNT='0')
    if token:
        basic = b64encode(('oauth2:'+token).encode()).decode('ascii')
        env.update(GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0=f'http.{ENDPOINT}/.extraHeader',
                   GIT_CONFIG_VALUE_0='Authorization: Basic '+basic)
    result = subprocess.run(['git','-c','credential.helper=','-c','http.followRedirects=false',
                             'ls-remote','--symref','--exit-code',url],
                            env=env,capture_output=True,text=True,timeout=45)
    if allow_empty and result.returncode == 2 and not result.stdout.strip():
        return None
    require(result.returncode == 0,'Could not read the ModelScope Git commit')
    branch = 'refs/heads/'+BRANCH
    default = f'ref: {branch}\tHEAD'
    lines = result.stdout.splitlines()
    require(default in lines,'Expected the registered ModelScope default branch')
    refs = dict(line.split('\t',1)[::-1] for line in lines if line != default)
    revision = refs.get(branch,'')
    require(len(lines) == 3 and set(refs) == {'HEAD',branch} and refs['HEAD'] == revision
            and re.fullmatch(r'[0-9a-f]{40}',revision),
            'Expected the single default branch of the release repository')
    return revision


def repo_identity(info, repo_id, license_id):
    owner,name = repo_id.split('/')
    require(info is not None and info.id and (info.owner,info.name) == (owner,name)
            and (info.license or '').lower() == license_id,
            'ModelScope repository identity or license differs from the approved package')


def inventory(hub, repo_id, revision):
    files = {}
    if revision is None:
        return files
    for item in hub.list_repo_files(repo_id,'model',revision=revision,recursive=True):
        if item.type == 'tree':
            continue
        name = item.path
        file_path(name)
        require(item.type == 'blob' and name not in files,'Unexpected ModelScope file entry')
        files[name] = {'bytes':item.size,'sha256':item.sha256}
    if files:
        validate_files(files)
    return files


def verify_private_download(hub, manifest):
    with tempfile.TemporaryDirectory(prefix='modelscope-private-readback-') as temp:
        for name,entry in manifest['files'].items():
            path = hub.download_file(manifest['repo_id'],'model',name,revision=manifest['revision'],local_dir=temp,
                                     force=True,expected_sha256=entry['sha256'],file_size=entry['bytes'])
            require(path.is_file() and path.stat().st_size == entry['bytes'] and sha256(path) == entry['sha256'],
                    'Private ModelScope readback differs from the approved bytes')


def stage(hub, directory, state_path, state, manifest, license_id):
    repo_id = manifest['repo_id']
    info = repository(hub,repo_id)
    if 'initial_files' not in state:
        if not state.get('creation_started'):
            require(info is None,'Choose an unused repository for this first release')
            state['creation_started'] = True
            atomic_json(state_path,state)
        if info is None:
            hub.create_repo(repo_id,'model',visibility='private',license=license_id)
            info = repository(hub,repo_id)
        repo_identity(info,repo_id,license_id)
        require(info.visibility == PRIVATE,'The newly created release repository must be private')
        initial = inventory(hub,repo_id,branch_head(repo_id,hub.legacy.token,allow_empty=True))
        require(set(initial) <= INITIAL_FILES,'Unexpected content in the newly created repository')
        state.update(repo_uid=info.id,initial_files=initial)
        atomic_json(state_path,state)

    repo_identity(info,repo_id,license_id)
    require(info.id == state['repo_uid'] and info.visibility == PRIVATE,
            'Require the same private repository recorded at initialization')
    payload = manifest['files']
    initial = state['initial_files']
    expected = initial | payload
    head = branch_head(repo_id,hub.legacy.token,allow_empty=True)
    current = inventory(hub,repo_id,head)
    if 'staged_manifest' in state:
        prior = state['staged_manifest']
        require(prior == dict(manifest,files=expected,revision=head) and current == expected,
                'The verified private commit changed')
        return prior
    require(all(value in (payload.get(name),initial.get(name)) for name,value in current.items()),
            'Existing repository bytes differ from the staged release')
    if current != expected:
        with tempfile.TemporaryDirectory(prefix='.modelscope-upload-',dir=directory) as temp:
            snapshot = Path(temp)
            for name in payload:
                relative = file_path(name)
                target = snapshot/relative
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(directory/'payload'/relative,target)
            verify_tree(snapshot,payload)
            hub.upload_folder(repo_id,'model',snapshot,revision=BRANCH,allow_patterns=sorted(payload),
                              commit_message='Upload the accepted 27B coding adapter',use_cache=False,
                              disable_tqdm=True,sync_remote_repo=False)
        info = repository(hub, repo_id)
        repo_identity(info, repo_id, license_id)
        require(info.id == state['repo_uid'] and info.visibility == PRIVATE,
                'The staged repository identity and privacy must survive the upload')
    head = branch_head(repo_id,hub.legacy.token)
    require(inventory(hub,repo_id,head) == expected,'Private commit inventory differs from the release package')
    pinned = dict(manifest,revision=head,files=expected)
    verify_private_download(hub,pinned)
    state['staged_manifest'] = pinned
    atomic_json(state_path,state)
    return pinned


def publish(hub, state_path, state, manifest, license_id, destination):
    require('staged_manifest' in state,'Verify the private staged commit before publishing')
    pinned = state['staged_manifest']
    repo_id = manifest['repo_id']
    info = repository(hub,repo_id)
    repo_identity(info,repo_id,license_id)
    require(info.id == state['repo_uid'] and info.visibility in (PRIVATE,PUBLIC)
            and pinned == dict(manifest,revision=pinned['revision'],files=state['initial_files'] | manifest['files'])
            and branch_head(repo_id,hub.legacy.token) == pinned['revision']
            and inventory(hub,repo_id,pinned['revision']) == pinned['files'],
            'Require the exact staged commit and repository before publication')
    existing = destination.exists() or destination.is_symlink()
    if 'public_readback' in state:
        require(destination.resolve() == Path(state['download_directory']),'Use the recorded public download directory')
        require(branch_head(repo_id) == pinned['revision'],'Public Git head differs from the release commit')
        if existing:
            verify_tree(destination,pinned['files'])
        else:
            require(readback(pinned,destination) == state['public_readback'],
                    'The restored download differs from the recorded ModelScope receipt')
        return state['public_readback']
    require(destination.parent.is_dir() and not destination.is_symlink(),
            'Choose a normal public download directory')
    if existing:
        require(info.visibility == PUBLIC,'An existing download can only be recovered for a public release')
        verify_tree(destination,pinned['files'])
    if info.visibility == PRIVATE:
        hub.update_repo_settings(repo_id,'model',visibility='public')
    require(branch_head(repo_id) == pinned['revision'],'Public Git head differs from the release commit')
    if existing:
        # An interruption can leave the completed download before its receipt is saved.
        with tempfile.TemporaryDirectory(prefix='.modelscope-public-retry-',dir=destination.parent) as temp:
            receipt = readback(pinned,Path(temp)/'snapshot')
    else:
        receipt = readback(pinned,destination)
    state.update(public_readback=receipt,download_directory=str(destination.resolve()))
    atomic_json(state_path,state)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','stage','publish'))
    parser.add_argument('--package',required=True,type=Path)
    parser.add_argument('--download-to',type=Path)
    args = parser.parse_args()
    try:
        manifest,license_id,digest = approved_package(args.package)
        repo_id = manifest['repo_id']
        if args.action == 'check':
            print(json.dumps({'repo_id':repo_id,'files_verified':len(manifest['files']),'eligible':True}))
            return

        hub = publishing_hub()
        require(hub.legacy.endpoint == ENDPOINT and bool(hub.legacy.token),
                'Sign in to the authorized ModelScope international account')
        require(hub.whoami().username.casefold() == OWNER.casefold(),
                'ModelScope login does not match the release account')
        state_path = args.package/'modelscope-release.json'
        expected = {'schema_version':1,'release_evidence_sha256':digest,'repo_id':repo_id,'endpoint':ENDPOINT}
        with single_writer(args.package/'.modelscope-release.lock'):
            state = read(state_path) if state_path.exists() else dict(expected)
            require(all(state.get(k) == v for k,v in expected.items()),'Release receipt belongs to a different package')
            if args.action == 'stage':
                result = stage(hub,args.package,state_path,state,manifest,license_id)
                print(json.dumps({'repo_id':repo_id,'revision':result['revision'],'private_readback_verified':True}))
            else:
                require(args.download_to is not None,'Provide --download-to for public verification')
                print(json.dumps(publish(hub,state_path,state,manifest,license_id,args.download_to)))
    except (ValueError,FileNotFoundError) as error:
        parser.exit(2,(str(error) if isinstance(error,ValueError) else 'A required release file is missing.')+'\n')
    except Exception as error:
        parser.exit(2,'ModelScope release step did not complete ('+type(error).__name__+').\n')


if __name__ == '__main__':
    main()
