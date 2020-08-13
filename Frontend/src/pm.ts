const fs = require('fs');
const { app, nativeTheme, BrowserWindow } = require('electron');

let mainWindow;
const createWindow = () => {
  // Create a window instance
  mainWindow = new BrowserWindow({
    width: 540,
    minWidth: 540,
    height: 540,
    minHeight: 480,
    show: false,
    frame: false,
    resizable: false,
    backgroundColor: nativeTheme.shouldUseDarkColors ? '#2b2b2b' : '#fff',
    // icon: './src/pm.png',
    webPreferences: { nodeIntegration: true },
  });

  // Load page
  console.log('Running Pixel Mazer.');
  const { port } = JSON.parse(fs.readFileSync('./package.json').toString());
  mainWindow.loadURL(`http://localhost:${port}/index.html`).then(() => {
    // Call loaded event in the window
    mainWindow.webContents.executeJavaScript(`PMSolver.setDevPort(${port});`);
  });

  // Bind events
  mainWindow.once('closed', () => { mainWindow = null; });
  mainWindow.once('ready-to-show', () => { mainWindow.show(); });
  mainWindow.on('focus', () => {
    // Call focusing event in the window
    mainWindow.webContents.executeJavaScript('PMSolver.setFocus(true);');
  });
  mainWindow.on('blur', () => {
    // Call blurring event in the window
    mainWindow.webContents.executeJavaScript('PMSolver.setFocus(false);');
  });

  // Call PM Projects
};

app.on('ready', createWindow);
app.on('window-all-closed', () => {
  // On macOS it is common for applications and their menu bar
  // to stay active until the user quits explicitly with Cmd + Q
  if (process.platform !== 'darwin') app.quit();
});
app.on('activate', () => {
  // On macOS it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (mainWindow === null) createWindow();
});
