const ipc = require('node-ipc');
const pm = require('./x64/Release/PM_C');

ipc.config.id = 'c_world';
ipc.config.retry = 1500;
ipc.config.silent = true;
ipc.serve(() => {
  console.log('Serving...');
  ipc.server.on('work', (doom) => {
    console.log('Received a map!');
    const doomer = pm.Solver(doom);
    doomer.onIteration(() => {
      ipc.server.broadcast('proc', doomer.getMapProc());
      console.log('Iteration reported.');
    });
    doomer.onSolved(() => {
      ipc.server.broadcast('solved', [doomer.getPerformanceReport()[0],
        null/* Solution will be in here */]);
      console.log('Solution reported.');
    });
    doomer.solveWithC();
  });
});
ipc.server.start();
