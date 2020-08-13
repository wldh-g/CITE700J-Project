import React, { useEffect, useState } from 'react';
import { PrimaryButton, DefaultButton, Text } from '@fluentui/react';
import colormap from 'colormap';

import './style.scss';

let scale = 1;
let sWidth = 1;
let sHeight = 1;
let startPoint = [0, 0];
let setMsgExternal = null;
const { remote } = window.module.require('electron');
const ipc = remote.require('node-ipc');
ipc.config.id = 'c_viewer';
ipc.config.retry = 1500;
ipc.config.silent = true;
ipc.connectTo(
  'c_world',
  () => {
    ipc.of.c_world.on('connect', () => {
      console.log('Connected to the GUI.');
    });
    ipc.of.c_world.on('disconnect', () => {
      console.log('Disconnected from the GUI.');
    });
    ipc.of.c_world.on(
      'proc',
      (data, socket) => {
        const [cnt, proc] = data;
        const cmap = colormap({
          colormap: 'jet',
          nshades: cnt >= 6 ? cnt : 6,
          format: 'hex',
          alpha: 1,
        });
        const canvas = document.getElementById('yeah') as HTMLCanvasElement;
        const ctx = canvas.getContext('2d');
        for (let y = 0; y < sHeight; y += 1) {
          for (let x = 0; x < sWidth; x += 1) {
            ctx.fillStyle = cmap[proc[y][x]];
            if (proc[y][x]) {
              ctx.fillRect(x * scale, y * scale, scale, scale);
            }
          }
        }
        console.log(data);
      },
    );
    ipc.of.c_world.on(
      'solved',
      (data, socket) => {
        const [perfReport, solution] = data;
        if (setMsgExternal != null) {
          setMsgExternal(`Solved in ${perfReport} ms with C impl.`);
        }
      },
    );
  },
);

interface Props {
  map: boolean[][];
  start: number[];
  end: number[];
  width: number;
  height: number;
}

const CSolver : React.SFC<Props> = (props: Props) => {
  const {
    map, start, end, width, height,
  } = props;
  scale = document.body.offsetWidth / width;
  sWidth = width;
  sHeight = height;
  startPoint = start;

  const [msg, setMsg] = useState('');
  setMsgExternal = setMsg;

  const sendPump = () => {
    ipc.of.c_world.emit('work', {
      map, start, end, width, height,
    });
  };

  const reset = () => {
    setMsg('Initializing...');
    const canvas = document.getElementById('yeah') as HTMLCanvasElement;
    const ctx = canvas.getContext('2d');
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        if (map[y][x]) {
          ctx.fillStyle = 'rgb(0, 0, 0)';
        } else {
          ctx.fillStyle = 'rgb(255, 255, 255)';
        }
        ctx.fillRect(x * scale, y * scale, scale, scale);
      }
    }
    setMsg('');
  };

  useEffect(() => { reset(); }, []);

  return (
    <>
      <div styleName="goodboy">
        <canvas id="yeah" width={document.body.offsetWidth} height={height * scale} />
      </div>
      <div styleName="badboy">
        <PrimaryButton onClick={sendPump}>Solve</PrimaryButton>
        <DefaultButton onClick={reset}>Reset</DefaultButton>
        <Text variant="xLarge">{msg}</Text>
      </div>
    </>
  );
};

export default CSolver;
