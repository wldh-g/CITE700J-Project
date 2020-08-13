import React from 'react';
import ReactDOM from 'react-dom';
import { loadTheme } from '@fluentui/react';

import TitleBar from './TitleBar';
import Navigation from './Navigation';
import * as theme from './theme';

import './main.scss';

// Add theming features
declare global { interface Window { PMSolver: any; } }
window.PMSolver = {};
window.PMSolver.setDevPort = (port: number) : void => {
  window.PMSolver.port = port;
};
window.PMSolver.applyPalette = () : void => {
  loadTheme(theme[document.body.dataset.theme]);
};
window.PMSolver.changeTheme = () : void => {
  if (document.body.dataset.theme === 'dark') {
    document.body.dataset.theme = 'light';
  } else {
    document.body.dataset.theme = 'dark';
  }
  window.PMSolver.applyPalette();
};
window.PMSolver.setFocus = (isFocused: boolean) : void => {
  if (isFocused) {
    document.body.dataset.focus = 'in';
  } else {
    document.body.dataset.focus = 'out';
  }
};

// Set main theme
const { remote } = window.module.require('electron');
document.body.dataset.theme = remote.nativeTheme.shouldUseDarkColors ? 'dark' : 'light';
window.PMSolver.applyPalette();

ReactDOM.render(
  <>
    <TitleBar showLabel />
    <Navigation />
  </>,
  document.getElementById('clientContainer'),
);
