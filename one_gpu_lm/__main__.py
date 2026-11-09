"""One GPU LM operational entry point; implementations remain in their owning modules."""
import argparse
import importlib
import sys


COMMANDS={
    'doctor':'one_gpu_lm.training.doctor',
    'train':'one_gpu_lm.training.runner',
    'status':'one_gpu_lm.training.status',
    'collect':'one_gpu_lm.training.collect',
    'verify-resume':'one_gpu_lm.training.verify_resume',
    'evaluate':'one_gpu_lm.evaluation.training_artifact',
    'serve':'one_gpu_lm.serving.local_backend',
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=COMMANDS)
    if len(sys.argv)>1 and sys.argv[1] in COMMANDS:
        module=COMMANDS[sys.argv[1]];sys.argv=[module,*sys.argv[2:]]
        importlib.import_module(module).main()
    else:p.parse_args()


if __name__=='__main__':main()
