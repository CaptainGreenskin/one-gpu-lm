"""Read a public ModelScope revision into a fresh directory and verify its files."""
import argparse
from hashlib import sha256
from http.client import HTTPException
import json
from pathlib import Path, PurePosixPath
import re
from stat import S_ISREG
import tempfile
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


OFFICIAL_SITES = {
    'https://www.modelscope.ai', 'https://modelscope.ai',
    'https://modelscope.cn', 'https://www.modelscope.cn',
}


class TLSRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        target = urlsplit(newurl)
        if target.scheme != 'https' or target.username is not None or target.password is not None:
            raise ValueError('ModelScope readback requires HTTPS redirects without user information')
        return super().redirect_request(request,fp,code,msg,headers,newurl)


def model_endpoint(endpoint, repo_id):
    endpoint = endpoint.rstrip('/')
    if endpoint not in OFFICIAL_SITES:
        raise ValueError('Specify the official ModelScope site for this release')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*',repo_id):
        raise ValueError('Expected a ModelScope owner/repository identifier')
    return endpoint+'/api/v1/models/'+repo_id


def file_path(name):
    if (not isinstance(name,str) or not name or '\\' in name
            or any(ord(char)<32 for char in name)
            or any(part in ('','.','..') for part in name.split('/'))
            or PurePosixPath(name).is_absolute()):
        raise ValueError('Expected a canonical path within the published repository')
    return PurePosixPath(name)


def validate_files(files):
    if not isinstance(files,dict) or not files:
        raise ValueError('Provide an independently verified release file manifest')
    seen = set()
    for name,entry in files.items():
        path = file_path(name)
        local_name = path.as_posix().casefold()
        if (not isinstance(entry,dict) or set(entry)!={'bytes','sha256'} or local_name in seen
                or type(entry['bytes']) is not int or entry['bytes']<0):
            raise ValueError('Invalid or ambiguous file manifest')
        if not isinstance(entry['sha256'],str) or not re.fullmatch(r'[0-9a-f]{64}',entry['sha256']):
            raise ValueError('Each release file needs a full SHA-256 digest')
        seen.add(local_name)


def read_json(opener, url):
    try:
        with opener.open(Request(url,headers={'Accept':'application/json'}),timeout=35) as response:
            returned_host = urlsplit(response.geturl()).hostname or ''
            value = json.load(response)
    except (OSError,HTTPException,ValueError) as error:
        status = getattr(error,'code',None) or type(error).__name__
        raise RuntimeError(f'Anonymous ModelScope metadata request failed ({status})') from None
    requested_host = urlsplit(url).hostname or ''
    if returned_host.removeprefix('www.') != requested_host.removeprefix('www.'):
        raise ValueError('ModelScope metadata moved to a different site')
    if value.get('Success') is not True or not isinstance(value.get('Data'),dict):
        raise ValueError('ModelScope did not confirm the anonymous request')
    return value['Data']


def published_files(opener, api, revision):
    query = urlencode({'Revision':revision,'Recursive':'True'})
    entries = read_json(opener,api+'/repo/files?'+query)['Files']
    files = {}
    for entry in entries:
        if entry['Type']=='tree':
            continue
        name = entry['Path']
        file_path(name)
        mode = str(entry['Mode'])
        if entry['Type']!='blob' or not S_ISREG(int(mode,8 if len(mode)==6 else 10)) or name in files:
            raise ValueError('The public inventory contains a duplicate or non-regular file')
        files[name] = {'bytes':entry['Size'],'sha256':entry['Sha256']}
    validate_files(files)
    return files


def download_file(opener, api, revision, name, entry, destination):
    query = urlencode({'Revision':revision,'FilePath':name})
    target = destination/file_path(name)
    target.parent.mkdir(parents=True,exist_ok=True)
    digest = sha256()
    size = 0
    try:
        with opener.open(Request(api+'/repo?'+query),timeout=45) as response, target.open('xb') as stream:
            for chunk in iter(lambda:response.read(1024*1024),b''):
                size += len(chunk)
                if size>entry['bytes']:
                    raise ValueError(f'Public file exceeds its manifest size: {name}')
                digest.update(chunk)
                stream.write(chunk)
    except (OSError,HTTPException) as error:
        status = getattr(error,'code',None) or type(error).__name__
        raise RuntimeError(f'Anonymous download failed for {name} ({status})') from None
    if size!=entry['bytes'] or digest.hexdigest()!=entry['sha256']:
        raise ValueError(f'Public file differs from the release manifest: {name}')


def readback(manifest, destination):
    endpoint,repo_id,revision = (manifest[k] for k in ('endpoint','repo_id','revision'))
    if not re.fullmatch(r'[0-9a-f]{40}',revision):
        raise ValueError('Pin the full ModelScope Git commit')
    api = model_endpoint(endpoint,repo_id)
    expected = manifest['files']
    validate_files(expected)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Choose a fresh download directory')

    # A new opener has no ModelScope cookies, authenticated SDK, or file cache.
    opener = build_opener(TLSRedirects())
    info = read_json(opener,api)
    if (info.get('Path'),info.get('Name')) != tuple(repo_id.split('/')) or info.get('Visibility')!=5:
        raise ValueError('ModelScope did not confirm the intended public repository')
    if published_files(opener,api,revision)!=expected:
        raise ValueError('Public ModelScope inventory differs from the release manifest')

    with tempfile.TemporaryDirectory(prefix='.modelscope-readback-',dir=destination.parent) as temporary:
        staged = Path(temporary)/'snapshot'
        staged.mkdir(mode=0o700)
        for name,entry in sorted(expected.items()):
            download_file(opener,api,revision,name,entry,staged)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError('Choose a fresh download directory')
        staged.rename(destination)
    return {'endpoint':endpoint,'repo_id':repo_id,'revision':revision,
            'model_url':endpoint.rstrip('/')+'/models/'+repo_id,
            'anonymous_public_readback_verified':True,
            'verified_files':len(expected),'verified_bytes':sum(entry['bytes'] for entry in expected.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    print(json.dumps(readback(manifest,args.out),indent=2))


if __name__ == '__main__':
    main()
