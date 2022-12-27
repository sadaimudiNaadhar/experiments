'use strict';

const { spawnSync } = require( 'child_process' );
const runExeCommand = spawnSync( 'main' );

console.log( `stderr: ${ runExeCommand.stderr.toString() }` );
console.log( `stdout: ${ runExeCommand.stdout.toString() }` );