"""Use a fresh comparison's measured candidate with the original one-attempt release suite."""
import argparse
import json
import os
from pathlib import Path
import signal

from one_gpu_lm.models.release_lineage import reviewed_training

from final_audit import audit_approved
from final_campaign import FINAL, interrupted, run_approved
from promotion import authorize_comparison
from release_package import assemble_approved


def advance(args):
    permission = authorize_comparison(args.protocol,args.controller,args.audit,args.publication,args.openings,
                                      args.seal,args.input_review)
    if args.action == 'check' or permission['approved'] is not True:
        return {key:permission[key] for key in ('approved','stage','final_test_opened','weights_public')}
    if args.action == 'score':
        profile, _ = reviewed_training(permission,args.input_review,args.training_protocol,args.training_profile,
                                      args.training_integrity)
        return run_approved(args.sealed.resolve(),args.adapter,permission,adapter_profile=profile)
    if args.action == 'audit':
        evidence = audit_approved(args.sealed.resolve(),permission)
        return {key:evidence[key] for key in ('execution_audit_passed','final_test_opened','weights_public')}
    manifest = assemble_approved(args,permission,args.sealed.resolve()/FINAL)
    return {'accepted':True,'staged_files':len(manifest['files']),'final_test_opened':True,'weights_public':False}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','controller','audit','publication','openings','seal','input-review'):
        parser.add_argument('--'+name,required=True,type=Path)
    phases = parser.add_subparsers(dest='action',required=True)
    phases.add_parser('check',help='Recompute the development decision and verify the exact promotion')
    score = phases.add_parser('score',help='Use the approved checkpoint in the original sealed suite')
    audit = phases.add_parser('audit',help='Replay the completed sealed suite for the promoted checkpoint')
    package = phases.add_parser('package',help='Stage the weights and model card after the release gate')
    for phase in (score,audit,package):
        phase.add_argument('--sealed',required=True,type=Path)
    for phase in (score,package):
        for name in ('adapter','training-protocol','training-profile','training-integrity'):
            phase.add_argument('--'+name,required=True,type=Path)
    for name in ('rights','notice-readback','notices','before','after','out'):
        package.add_argument('--'+name,required=True,type=Path)
    package.add_argument('--repo-id',required=True)
    return parser.parse_args()


if __name__ == '__main__':
    os.umask(0o077)
    args = arguments()
    if args.action == 'score':
        for kind in (signal.SIGTERM,signal.SIGINT):
            signal.signal(kind,interrupted)
    print(json.dumps(advance(args)))
