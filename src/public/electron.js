const { app, BrowserWindow, ipcMain, dialog, protocol } = require('electron');
const path = require('path');
const isDev = require('electron-is-dev');
const { PythonShell } = require('python-shell');
const fs = require('fs').promises;
const { spawn } = require('child_process');

let mainWindow;

async function getConfigPath() {
  return path.join(app.getPath('userData'), 'config.json');
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  mainWindow.loadURL(
    isDev
      ? 'http://localhost:3000'
      : `file://${path.join(__dirname, '../build/index.html')}`
  );

  if (isDev) {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

async function loadConfig() {
  const configPath = await getConfigPath();
  try {
    const data = await fs.readFile(configPath, 'utf8');
    return JSON.parse(data);
  } catch (error) {
    if (error.code === 'ENOENT') {
      return {
        pre_directory_location: '',
        post_directory_location: '',
        nuclei_model_location: '',
        cyto_model_location: '',
        experiment_name: 'Default_Experiment'
      };
    }
    throw error;
  }
}

async function saveConfig(config) {
  const configPath = await getConfigPath();
  await fs.writeFile(configPath, JSON.stringify(config, null, 2));
}

app.whenReady().then(() => {
  protocol.registerFileProtocol('file', (request, callback) => {
    const url = request.url.substr(8);
    callback({ path: decodeURI(url) });
  });

  createWindow();

  ipcMain.handle('run-python-script', async (event, scriptName) => {
    return new Promise(async (resolve, reject) => {
      const pythonScriptPath = path.join(__dirname, '..', 'public', 'python_scripts', scriptName);
      const configPath = await getConfigPath();

      console.log('Python script path:', pythonScriptPath);
      console.log('Config path:', configPath);

      const pythonProcess = spawn('python', [pythonScriptPath, configPath]);

      let output = '';
      pythonProcess.stdout.on('data', (data) => {
        if (mainWindow) {
          mainWindow.webContents.send('python-output', data.toString());
        }
        output += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        console.error(`Python script error: ${data}`);
        reject(data.toString());
      });

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          resolve(output);
        } else {
          reject(`Python script exited with code ${code}`);
        }
      });
    });
  });

  ipcMain.handle('get-graph-paths', async () => {
    const configPath = await getConfigPath();
    const config = JSON.parse(await fs.readFile(configPath, 'utf8'));
    const graphDir = path.resolve(app.getPath('userData'), config.experiment_name);
    console.log('Looking for graphs in:', graphDir);
    const files = await fs.readdir(graphDir);
    const graphFiles = files.filter(file => file.endsWith('.png'));
    
    const graphData = await Promise.all(graphFiles.map(async (file) => {
      const filePath = path.join(graphDir, file);
      const data = await fs.readFile(filePath);
      return {
        name: file,
        data: `data:image/png;base64,${data.toString('base64')}`
      };
    }));

    console.log(`Loaded ${graphData.length} graph images`);
    return graphData;
  });

  ipcMain.handle('load-config', loadConfig);

  ipcMain.handle('save-config', async (event, config) => {
    await saveConfig(config);
    return true;
  });

  ipcMain.handle('open-folder-dialog', async () => {
    const result = await dialog.showOpenDialog({
      properties: ['openDirectory']
    });
    return result.canceled ? null : result.filePaths[0];
  });

  ipcMain.handle('open-file-dialog', async () => {
    const result = await dialog.showOpenDialog({
      properties: ['openFile']
    });
    return result.canceled ? null : result.filePaths[0];
  });
});

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