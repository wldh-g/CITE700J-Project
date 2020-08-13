import React from 'react';
import ReactDOM from 'react-dom';

import { loadTheme } from '@fluentui/react';

import CUDASolver from './CUDASolver';
import TitleBar from './TitleBar';

import * as theme from './theme';
import './main.scss';

// Set main theme
document.body.dataset.theme = 'dark';
loadTheme(theme.dark);

// eslint-disable-next-line @typescript-eslint/no-unused-vars
declare global { interface Window { PMSolver: any; } }
window.PMSolver = {};
window.PMSolver.launchRendering = (doomstr) => {
  const doom = JSON.parse(doomstr);

  // Render main DOM
  ReactDOM.render(
    <>
      <TitleBar />
      <CUDASolver
        map={doom.map}
        start={doom.start}
        end={doom.end}
        width={doom.width}
        height={doom.height}
      />
    </>,
    document.getElementById('clientContainer'),
  );
};
