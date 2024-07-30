const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const isDev = require('electron-is-dev');
const { PythonShell } = require('python-shell');
const fs = require('fs').promises;

function createWindow() {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  win.loadURL(
    isDev
      ? 'http://localhost:3000'
      : `file://${path.join(__dirname, '../build/index.html')}`
  );

  if (isDev) {
    win.webContents.openDevTools();
  }
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

ipcMain.handle('run-python-script', async (event, scriptName, args) => {
  return new Promise((resolve, reject) => {
    PythonShell.run(
      path.join(__dirname, 'python_scripts', scriptName),
      { args: args },
      function (err, results) {
        if (err) reject(err);
        resolve(results);
      }
    );
  });
});

ipcMain.handle('open-folder-dialog', async () => {
  const result = await dialog.showOpenDialog({
    properties: ['openDirectory']
  });
  return result.canceled ? null : result.filePaths[0];
});

function getSettingsPath() {
  return path.join(app.getPath('userData'), 'settings.json');
}

ipcMain.handle('save-settings', async (event, settings) => {
  try {
    await fs.writeFile(getSettingsPath(), JSON.stringify(settings, null, 2));
    return true;
  } catch (error) {
    console.error('Failed to save settings:', error);
    return false;
  }
});

ipcMain.handle('load-settings', async () => {
  try {
    const data = await fs.readFile(getSettingsPath(), 'utf8');
    return JSON.parse(data);
  } catch (error) {
    console.error('Failed to load settings:', error);
    return null;
  }
});