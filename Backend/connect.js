const ipc = require('node-ipc');
const pm = require('./x64/Release/PM_CUDA');

const c_ipc = new ipc.IPC;
c_ipc.config.id = 'c_world';
c_ipc.config.retry = 1500;
c_ipc.config.silent = true;
c_ipc.serve(() => {
  console.log('   C : Serving...');
  c_ipc.server.on('work', (doom) => {
    console.log('   C : Received a map!');
    const doomer = pm.Solver(doom);
    doomer.onIteration(() => {
      c_ipc.server.broadcast('proc', doomer.getMapProc());
      console.log('   C : Iteration reported.');
    });
    doomer.onSolved(() => {
      c_ipc.server.broadcast('solved', [doomer.getPerformanceReport()[0],
        null/* Solution will be in here */]);
      console.log('   C : Solution reported.');
    });
    doomer.solveWithC();
  });
});
c_ipc.server.start();

const cuda_ipc = new ipc.IPC;
cuda_ipc.config.id = 'cuda_world';
cuda_ipc.config.retry = 1500;
cuda_ipc.config.silent = true;
cuda_ipc.serve(() => {
  console.log('CUDA : Serving...');
  cuda_ipc.server.on('work', (doom) => {
    console.log('CUDA : Received a map!');
    const doomer = pm.Solver(doom);
    doomer.onIteration(() => {
      cuda_ipc.server.broadcast('proc', doomer.getMapProc());
      console.log('CUDA : Iteration reported.');
    });
    doomer.onSolved(() => {
      cuda_ipc.server.broadcast('solved', [doomer.getPerformanceReport()[0],
        null/* Solution will be in here */]);
      console.log('CUDA : Solution reported.');
    });
    doomer.solveWithCUDA();
  });
});
cuda_ipc.server.start();
