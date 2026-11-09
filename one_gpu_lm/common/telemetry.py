"""Per-attempt GPU measurements with UTC timestamps and explicit collection failures."""
import csv
from datetime import datetime, timezone
import subprocess
import threading
import time


class Telemetry:
    def __init__(self, path, attempt, interval=5):
        self.path,self.attempt,self.interval=path,attempt,interval
        self.stop_event=threading.Event()
        self.errors=0
        self.samples=0
        self.thread=threading.Thread(target=self.loop,daemon=True)

    def loop(self):
        fresh=not self.path.exists()
        with self.path.open('a') as stream:
            writer=csv.writer(stream)
            if fresh:
                writer.writerow(['utc','attempt','temperature_c','power_w','utilization_percent','memory_used_mib'])
            while True:
                try:
                    raw=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,timeout=4)
                    numbers=[float(v.strip()) for v in raw.strip().split(',')]
                    if len(numbers)!=4:
                        raise ValueError('Unexpected GPU sample')
                    writer.writerow([datetime.now(timezone.utc).isoformat(),self.attempt,*numbers])
                    stream.flush();self.samples+=1
                except (OSError,ValueError,subprocess.SubprocessError):
                    self.errors+=1
                if self.stop_event.wait(self.interval):
                    break

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set();self.thread.join(timeout=6)
        return {'samples':self.samples,'errors':self.errors,'thread_stopped':not self.thread.is_alive()}
