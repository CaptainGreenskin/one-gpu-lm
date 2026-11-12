"""One development-only repeated evaluation; no training,final tests or uploads."""
import argparse,json,os,signal,subprocess,sys,tempfile,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import atomic_json,single_writer,gpu_lock_path,utc
from one_gpu_lm.common.source import campaign_sources
from one_gpu_lm.common.telemetry import Telemetry
from one_gpu_lm.serving.client import request
from normalize import digest

def terminate(p):
 if p is not None and p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)

def execute(root,out,resume=False):
 root=Path(root).resolve();out=Path(out).resolve();os.chdir(root)
 with single_writer(root/'private/controller023.lock'),single_writer(gpu_lock_path()):
  return locked(root,out,resume)

def locked(root,out,resume):
 exp=root/'experiments/023-evaluation-stability';read=lambda p:json.loads(Path(p).read_text())
 protocol=read(exp/'protocol.json');plan=read(exp/'plan.json');runtime=read(root/'runtime.json')
 assert protocol['status']=='frozen_before_model_rollouts' and not protocol['final_test_opened']
 for name,h in protocol['frozen_files'].items():
  if sha256(root/name)!=h:raise ValueError('Changed frozen input: '+name)
 pub=read(root/'private/public-preregistration.json')
 assert pub['anonymous_files_verified'] and pub['ci_conclusion']=='success' and pub['protocol_sha256']==sha256(exp/'protocol.json')
 assert pub['commit']==read(root/'launch.json')['source_commit']
 tasks=read(root/'private/tasks.json')['tasks'];ids=[t['id'] for t in tasks]
 assert len(tasks)==15 and all(t['split']=='dev' for t in tasks)
 if out.exists()!=resume:raise ValueError('Explicit resume required only for existing run')
 if resume:
  state=read(out/'controller.json');assert not state['complete'] and state['protocol_sha256']==sha256(exp/'protocol.json')
  atomic_json(out/'private'/f'controller-attempt-{state["attempt"]}.json',state);state['attempt']+=1;state.pop('error_type',None)
 else:
  out.mkdir(parents=True,mode=0o700);(out/'private').mkdir(mode=0o700);(out/'probes').mkdir();tick=time.time()
  state={'schema_version':1,'started_utc':utc(),'started_epoch':tick,'deadline_epoch':tick+plan['budget']['max_pipeline_wall_seconds'],
   'protocol_sha256':sha256(exp/'protocol.json'),'complete':False,'attempt':1,'completed_blocks':[],
   'final_test_opened':False,'training_updates':0,'weights_public':False}
  atomic_json(out/'controller.json',state)
 deadline=time.monotonic()+max(0,state['deadline_epoch']-time.time());attempt=state['attempt']
 env=dict(os.environ,**read(root/'private/inputs.json'),CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
 cpu=str(root/'.venv-agent/bin/python');gpu=runtime['model_python']
 def remaining():
  left=deadline-time.monotonic()
  if left<=0:raise TimeoutError('Original16-hour deadline exhausted')
  return left
 def stage(name,**fields):
  state.update(stage=name,updated_utc=utc(),**fields);atomic_json(out/'controller.json',state);print(name,flush=True)
 def command(label,args,seconds):
  p=None
  try:
   with (out/'private'/f'{label}-attempt-{attempt}.log').open('a') as log:
    p=subprocess.Popen(args,stdout=log,stderr=log,env=env,start_new_session=True)
    if p.wait(timeout=min(seconds,remaining())):raise RuntimeError('Stage failed: '+label)
  finally:terminate(p)
 def new_campaign(label,block,condition):
  folder=out/'campaigns'/label;p=folder/'campaign.json'
  identity={'tasks_sha256':sha256(root/'private/tasks.json'),'admission_sha256':protocol['admission_sha256'],
   'protocol_sha256':sha256(exp/'protocol.json'),'ids':ids,'variant':condition,'model':block['model'],'repeat':block['repeat'],
   'source_files':campaign_sources(),'wrapper_sha256':sha256(exp/'agent.py'),'normalizer_sha256':sha256(exp/'normalize.py')}
  if p.exists():
   value=read(p);assert value['identity']==identity
  else:
   folder.mkdir(parents=True);value={'schema_version':1,'started_utc':utc(),'identity':identity,'results':[],'complete':False};atomic_json(p,value)
  return folder,value
 def interrupted(signum,frame):raise KeyboardInterrupt('Controller interrupted')
 for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,interrupted)
 monitor=Telemetry(out/f'telemetry-attempt-{attempt}.csv',attempt,5);monitor.start()
 try:
  for block in protocol['blocks']:
   label=f'r{block["repeat"]:02d}-{block["model"]}'
   if label in state['completed_blocks']:continue
   cells={c:new_campaign(label+'-'+c,block,c) for c in plan['conditions']}
   model=block['model'];adapter=root/runtime['reference_adapter'] if model=='reference020' else root/runtime['action_adapter'] if model=='action022' else None
   hashes={'adapter_model.safetensors':plan['models'][model]['adapter_model_sha256']} if adapter else {}
   backend=None;backend_out=out/'private'/f'backend-{label}-attempt-{attempt}'
   assert not backend_out.exists()
   with tempfile.TemporaryDirectory(prefix='oglm023-') as tmp:
    sock=Path(tmp)/'model.sock'
    args=[gpu,str(exp/'serve.py'),'--model',runtime['model'],'--profile',str(exp/'profile.json'),'--socket',str(sock),'--out',str(backend_out)]
    if adapter:args+=['--adapter',str(adapter)]
    stage('loading_'+label,block=label)
    with (out/'private'/f'backend-{label}-attempt-{attempt}.log').open('x') as log:
     try:
      backend=subprocess.Popen(args,env=env,stdout=log,stderr=log,start_new_session=True)
      ready=time.monotonic()+min(plan['budget']['max_backend_start_seconds'],remaining())
      while not (backend_out/'ready.json').exists():
       if backend.poll() is not None:raise RuntimeError('Backend exited before ready')
       if time.monotonic()>ready:raise TimeoutError('Backend readiness timed out')
       time.sleep(1)
      health=request(sock,{'operation':'health'})
      assert health['profile_sha256']==sha256(exp/'profile.json') and health['adapter_sha256']==hashes
      atomic_json(out/f'artifact-{label}-attempt-{attempt}.json',health)
      controls=out/'loaded-controls'/f'{label}-attempt-{attempt}'
      stage('controls_'+label)
      command('controls-'+label,[cpu,'-m','one_gpu_lm.evaluation.preflight','--root',str(root),'--tasks',str(root/'private/tasks.json'),'--out',str(controls),'--clean-only'],600)
      actual=read(controls/'admission.json');assert actual['complete'] and len(actual['clean'])==4 and all(r['passed'] for r in actual['clean'])
      for probe in read(root/'private/probes.json'):
       for repeat in range(2):
        name=f'{label}-{probe["id"]}-{repeat+1}';record=out/'probes'/(name+'.json')
        if record.exists():continue
        stage('probe_'+name)
        payload={'operation':'generate','messages':probe['messages'],'max_new_tokens':512,'max_input_tokens':8192,'agent_policy':plan['policy']}
        # Preserve an in-flight marker so interruption cannot silently rerun a probe.
        pending=out/'private'/('probe-'+name+'.pending')
        if pending.exists():raise ValueError('Interrupted probe must remain incomplete, not regenerated')
        pending.write_text(utc()+'\n');response=request(sock,payload,timeout=min(180,remaining()))
        assert response['input_tokens_sha256']==probe['expected_input_sha256'] and response['input_tokens']==probe['expected_input_tokens']
        atomic_json(out/'private'/('probe-'+name+'.json'),response)
        public={k:v for k,v in response.items() if k not in {'text','ids'}}
        public.update(id=probe['id'],model=model,repeat=block['repeat'],within_load_repeat=repeat+1,
          output_sha256=digest(response['text']),ids_sha256=digest(json.dumps(response['ids'],separators=(',',':'))),backend_attempt=attempt)
        atomic_json(record,public);pending.unlink()
      for entry in block['tasks']:
       condition=entry['condition'];ident=entry['id'];folder,campaign=cells[condition]
       if any(r['id']==ident for r in campaign['results']):continue
       target=folder/'private'/ident;result=target/'result.json'
       stage('task',block=label,task_id=ident,condition=condition,completed_in_block=sum(len(c[1]['results']) for c in cells.values()))
       if not result.exists():
        args=[cpu,str(exp/'agent.py'),'--root',str(root),'--tasks',str(root/'private/tasks.json'),'--protocol',str(exp/'protocol.json'),
         '--socket',str(sock),'--out',str(target),'--id',ident,'--variant',condition]
        if (target/'checkpoint.json').exists():args+=['--resume']
        command(label+'-'+condition+'-'+ident,args,1020)
       row=read(result);assert row['id']==ident and row['split']=='dev' and row['output_condition']==condition
       if row['backend_errors']:raise RuntimeError('Backend error retained; do not score a replacement task')
       campaign['results'].append(row);campaign['complete']=len(campaign['results'])==15
       if campaign['complete']:campaign['finished_utc']=utc()
       atomic_json(folder/'campaign.json',campaign)
      assert all(c[1]['complete'] for c in cells.values())
      state['completed_blocks'].append(label);stage('block_completed',block=label)
     finally:terminate(backend)
  stage('completed',complete=True,finished_utc=utc(),finished_epoch=time.time())
 except BaseException as error:
  stage('failed',error_type=type(error).__name__,finished_utc=utc(),finished_epoch=time.time());raise
 finally:
  state['telemetry']=monitor.stop();atomic_json(out/'controller.json',state)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();execute(a.root,a.out,a.resume)
