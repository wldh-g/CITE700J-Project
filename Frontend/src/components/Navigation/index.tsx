import React, { useState } from 'react';
import { Text, PrimaryButton, DefaultButton } from '@fluentui/react';

import './style.scss';

const { remote } = window.module.require('electron');

const Navigation: React.FC = () => {
  const [msg, setMsg] = useState('Open the image to start maze solving.');
  const [ready, setReady] = useState(false);
  const [doom, setDoom] = useState({
    start: [3, 0],
    end: [31, 40],
    map: null,
    width: 0,
    height: 0,
  });

  // File management
  const openFile = () => {
    remote.dialog.showOpenDialog({
      properties: ['openFile'],
      filters: [{
        name: 'PNG Image',
        extensions: ['png'],
      }],
    }).then((result) => {
      setReady(false);
      if (!result.canceled) {
        setMsg('Manipulating pixels...');
        const sharp = remote.require('sharp');
        const image = sharp(result.filePaths[0]);
        image.metadata()
          .then((metadata) => {
            doom.map = new Array(metadata.height);
            doom.width = metadata.width;
            doom.height = metadata.height;
            return image
              .extractChannel('green')
              .toColourspace('b-w')
              .raw()
              .toBuffer();
          })
          .then((dimige) => {
            for (let y = 0; y < doom.height; y += 1) {
              const row = new Array(doom.width);
              for (let x = 0; x < doom.width; x += 1) {
                row[x] = dimige[x + y * doom.width] < 127;
              }
              doom.map[y] = row;
            }
            setMsg(`${doom.width} by ${doom.height} image loaded.`);
            setDoom(doom);
            setReady(true);
          })
          .catch((error) => {
            console.error(error);
          });
      }
    }).catch((err) => {
      console.error(err);
    });
  };

  // Window openers
  const openWithC = () => {
    let windowHeight = doom.height;
    let scaleFactor = 1;
    if (windowHeight > 920) {
      while (windowHeight > 920) {
        scaleFactor *= 0.9;
        windowHeight *= 0.9;
      }
    } else {
      while (windowHeight < 480) {
        scaleFactor *= 2;
        windowHeight *= 2;
      }
    }

    const windowWidth = doom.width * scaleFactor;
    let liveWindow = new remote.BrowserWindow({
      width: windowWidth,
      height: windowHeight + 100,
      backgroundColor: '#2b2b2b',
      show: false,
      frame: false,
      resizable: false,
      webPreferences: { nodeIntegration: true },
    });

    liveWindow.once('closed', () => {
      liveWindow = null;
    });

    if (window.PMSolver.port) {
      liveWindow.loadURL(`http://localhost:${window.PMSolver.port}/c.html`);
    } else {
      console.error('Port not found');
    }

    liveWindow.once('ready-to-show', () => {
      liveWindow.show();
      liveWindow.webContents.executeJavaScript(`PMSolver.launchRendering('${JSON.stringify(doom)}');`);
    });
  };

  const openWithCUDA = () => {
    let windowHeight = doom.height;
    let scaleFactor = 1;
    if (windowHeight > 920) {
      while (windowHeight > 920) {
        scaleFactor *= 0.9;
        windowHeight *= 0.9;
      }
    } else {
      while (windowHeight < 480) {
        scaleFactor *= 2;
        windowHeight *= 2;
      }
    }

    const windowWidth = doom.width * scaleFactor;
    let liveWindow = new remote.BrowserWindow({
      width: windowWidth,
      height: windowHeight + 100,
      backgroundColor: '#2b2b2b',
      show: false,
      frame: false,
      resizable: false,
      webPreferences: { nodeIntegration: true },
    });

    liveWindow.once('closed', () => {
      liveWindow = null;
    });

    if (window.PMSolver.port) {
      liveWindow.loadURL(`http://localhost:${window.PMSolver.port}/cuda.html`);
    } else {
      console.error('Port not found');
    }

    liveWindow.once('ready-to-show', () => {
      liveWindow.show();
      liveWindow.webContents.executeJavaScript(`PMSolver.launchRendering('${JSON.stringify(doom)}');`);
    });
  };

  return (
    <div styleName="client">
      <img src="./maze.svg" alt="Maze" />
      <div styleName="menu">
        <PrimaryButton onClick={openFile} styleName="image-button">
          Open Image
        </PrimaryButton>
        <Text variant="medium">{msg}</Text>
        <div styleName="navigation">
          <DefaultButton onClick={openWithC} disabled={!ready}>
            Solve with C
          </DefaultButton>
          <DefaultButton onClick={openWithCUDA} disabled={!ready}>
            Solve with CUDA
          </DefaultButton>
        </div>
        <br />
        <Text variant="smallPlus">20202132 Jio Gim.</Text>
      </div>
    </div>
  );
};

export default Navigation;
