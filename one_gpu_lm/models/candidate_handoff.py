"""Move the exact audited LoRA and its provenance to a private development workspace."""
import argparse
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.verified.tasks import relative
from .adapter_bundle import FILES, restore_adapter, verify_adapter
from .release_lineage import audited_training, read_digest


TRAINING_RECEIPTS = {'adapter.json','curve.json','environment.json','result.json','selection.json','updates.json'}


def ordinary_files(directory):
    files = list(directory.iterdir())
    if any(file.is_symlink() or not file.is_file() for file in files):
        raise ValueError('Checkpoint contains an unexpected file or directory')
    return {file.name:sha256(file) for file in files}


def family(name):
    return name.split(':',1)[0].casefold()


def stage(training_root, protocol_path, review_path, integrity_path, evaluation_profile_path,
          cohort_path, openings_path, destination):
    """Install one audited checkpoint alongside its exact private training evidence."""
    protocol,_ = read_digest(protocol_path)
    profile_path = training_root/relative(protocol['profile_path'])
    evidence = audited_training(review_path,protocol_path,profile_path,integrity_path)
    protocol,review,integrity = evidence.protocol,evidence.review,evidence.integrity
    run = training_root/relative(protocol['run_directory'])
    receipts = integrity['receipt_sha256']
    if (set(receipts) != TRAINING_RECEIPTS
            or any(sha256(run/relative(name))!=digest for name,digest in receipts.items())
            or sha256(training_root/relative(protocol['data_path']))!=evidence.lineage['training_data_sha256']
            or sha256(training_root/relative(protocol['sources_path']))!=evidence.lineage['source_manifest_sha256']):
        raise ValueError('Training results or source manifests differ from their audit')
    result,result_hash = read_digest(run/'result.json')
    if (result_hash!=receipts['result.json'] or result['complete'] is not True
            or result['all_selected_records_consumed'] is not True or result['saved_reload_greedy_ids_equal'] is not True
            or result['final_test_opened'] or result['weights_public'] or result['development_scores']!=0
            or result['protocol_sha256']!=evidence.lineage['protocol_sha256']
            or result['profile_sha256']!=evidence.lineage['profile_sha256']
            or result['data_sha256']!=evidence.lineage['training_data_sha256']
            or result['optimizer_steps']!=evidence.lineage['optimizer_updates']):
        raise ValueError('Expected one complete audited training attempt and matching reload probe')

    evaluation,eval_hash = read_digest(evaluation_profile_path)
    cohort,cohort_hash = read_digest(cohort_path)
    openings,openings_hash = read_digest(openings_path)
    keys = ('model_id','revision','loader','dtype','attention','chat_template_kwargs')
    cohort_keys = ('runtime_sha256','sources_sha256','tasks_sha256')
    if (any(evaluation[key]!=evidence.profile[key] for key in keys)
            or (evaluation['model_id'],evaluation['revision'])!=(openings['model_id'],openings['revision'])
            or openings['profile_sha256']!=eval_hash or openings['cohort_sha256']!=cohort_hash
            or any(openings[key]!=cohort[key] for key in cohort_keys)
            or openings['model_generations']!=0 or openings['optimizer_updates']!=0 or cohort['model_generations']!=0
            or openings['actor_count']!=cohort['cases'] or openings['final_test_opened'] or cohort['final_test_opened']):
        raise ValueError('The prepared development prompt, serving profile or training base differs')
    trained = {family(name) for name in review['sources']}
    if not trained or trained & {family(name) for name in cohort['families']}:
        raise ValueError('The candidate includes a development repository family in training')

    source = run/'private/adapter'
    files = review['checkpoint_files_sha256']
    if ordinary_files(source)!=files:
        raise ValueError('Checkpoint bundle differs from the completed training audit')
    weights = {name:files[name] for name in FILES}
    copies = {'training-protocol.json':(protocol_path,evidence.lineage['protocol_sha256']),
              'training-profile.json':(profile_path,evidence.lineage['profile_sha256']),
              'candidate-input-review.json':(review_path,evidence.review_sha256),
              'integrity.json':(integrity_path,evidence.lineage['integrity_sha256'])}
    report = {'schema_version':1,'training_lineage':evidence.lineage,'checkpoint_files_sha256':files,
              'candidate_review_sha256':evidence.review_sha256,'evaluation_profile_sha256':eval_hash,
              'development_opening_sha256':openings_hash,'development_cohort_sha256':cohort_hash,
              'cases':cohort['cases'],'training_source_families':len(trained),'model_generations':0,
              'cpu_tensor_check':True,'final_test_opened':False,'weights_public':False}

    def verify(directory):
        if (directory.is_symlink() or (directory/'adapter').is_symlink() or (directory/'handoff.json').is_symlink()
                or {p.name for p in directory.iterdir()} != set(copies)|{'adapter','handoff.json'}
                or ordinary_files(directory/'adapter')!=files
                or any((directory/name).is_symlink() or sha256(directory/name)!=digest
                       for name,(_,digest) in copies.items())
                or read_digest(directory/'handoff.json')[0]!=report):
            raise ValueError('Staged candidate or evidence differs from the verified handoff')
        verified = verify_adapter(directory/'adapter',evidence.profile,weights)
        if verified['parameters']!=result['trainable_parameters']:
            raise ValueError('Restored LoRA tensors differ from the recorded training capacity')
        return {**report,'adapter_parameters':verified['parameters']}

    if destination.exists():
        return verify(destination)
    if destination.is_symlink():
        raise FileExistsError('Choose a fresh private candidate destination')
    with TemporaryDirectory(prefix='.candidate-handoff-',dir=destination.parent) as temporary:
        staging = Path(temporary)/'bundle'
        staging.mkdir(mode=0o700)
        restore_adapter(source,staging/'adapter',evidence.profile,weights)
        for name in files.keys()-weights.keys():
            shutil.copyfile(source/relative(name),staging/'adapter'/relative(name))
        for name,(path,_) in copies.items():
            shutil.copyfile(path,staging/name)
        write_json(staging/'handoff.json',report)
        checked = verify(staging)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError('Candidate directory appeared during handoff')
        staging.rename(destination)
    return checked


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('training-root','protocol','review','integrity','evaluation-profile','cohort','openings','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    value = stage(args.training_root,args.protocol,args.review,args.integrity,args.evaluation_profile,args.cohort,
                  args.openings,args.out)
    print(json.dumps({'cases':value['cases'],'checkpoint_files':len(value['checkpoint_files_sha256']),
                      'adapter_parameters':value['adapter_parameters'],'final_test_opened':False,'weights_public':False}))


if __name__=='__main__':
    main()
