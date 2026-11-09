// Rebuild upstream's non-strict UMD runtime from the CURRENT authored sources.
import {createRequire} from 'node:module';
import Module from 'node:module';
const require = createRequire('/work/package.json');
const {rollup} = require('/work/node_modules/rollup');
const bundle = await rollup({input: '/work/modules/index-default.js', treeshake: false});
await bundle.write({file: '/work/underscore-umd.js', format: 'umd', name: '_',
  exports: 'default', noConflict: true, strict: false, freeze: false,
  externalLiveBindings: false, amd: {id: 'underscore'}});
await bundle.close();
const api = require('/work/underscore-umd.js');
const QUnit = require('/work/node_modules/qunit');
globalThis.QUnit = QUnit;
QUnit.config.autostart = false;
const originalLoad = Module._load;
Module._load = function(request, parent, ...rest) {
  if (request === '..' && parent?.filename.startsWith('/work/test/')) return api;
  return originalLoad.call(this, request, parent, ...rest);
};
for (const name of ['arrays', 'collections', 'objects', 'functions', 'utility', 'chaining']) {
  require(`/work/test/${name}.js`);
}
QUnit.on('testEnd', details => {
  if (details.status === 'failed') console.log(JSON.stringify({name: details.name, errors: details.errors}));
});
QUnit.on('runEnd', details => {
  console.log(JSON.stringify({tests: details.testCounts}));
  process.exitCode = details.testCounts.failed ? 1 : 0;
});
QUnit.start();
