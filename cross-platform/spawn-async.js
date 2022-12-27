'use strict';

const { spawn } = require( 'child_process' );
const runExeCommand = spawn( 'main');

runExeCommand.stdout.on( 'data', ( data ) => {
    console.log( `stdout: ${ data }` );
} );

runExeCommand.stderr.on( 'data', ( data ) => {
    console.log( `stderr: ${ data }` );
} );

runExeCommand.on( 'close', ( code ) => {
    console.log( `child process exited with code ${ code }` );
} );