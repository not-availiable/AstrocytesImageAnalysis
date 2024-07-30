import React, { useState, useEffect } from 'react';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';
import SettingsIcon from '@mui/icons-material/Settings';
import { styled, keyframes } from '@mui/system';
import { Box, Button, Typography, Paper, Grid, Modal, TextField, Fab, CircularProgress } from '@mui/material';

const { ipcRenderer } = window.require('electron');

const glowingOutline = keyframes`
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
`;

const GlowElement = styled(Box)(({ theme }) => ({
  borderRadius: '15px',
  transition: 'all 0.3s ease',
  position: 'relative',
  overflow: 'hidden',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: '-2px',
    left: '-2px',
    right: '-2px',
    bottom: '-2px',
    background: 'linear-gradient(90deg, #ff8a00, #ffc000, #ff8a00)',
    backgroundSize: '200% 200%',
    animation: `${glowingOutline} 3s ease-in-out infinite`,
    opacity: 0,
    transition: 'opacity 0.3s ease',
    borderRadius: '17px',
    zIndex: -1,
  },
  '&:hover::before': {
    opacity: 1,
  },
  '&:hover': {
    boxShadow: '0 0 20px rgba(255, 165, 0, 0.5)',
  },
}));

const StyledButton = styled(Button)(({ theme }) => ({
  borderRadius: '15px',
  background: 'rgba(255, 255, 255, 0.1)',
  backdropFilter: 'blur(5px)',
  transition: 'all 0.3s ease',
  '&:hover': {
    transform: 'scale(1.05)',
    background: 'rgba(255, 255, 255, 0.2)',
  },
}));

const StyledPaper = styled(Paper)(({ theme }) => ({
  background: 'rgba(255, 255, 255, 0.1)',
  backdropFilter: 'blur(10px)',
  padding: theme.spacing(2),
}));

const SettingsModal = styled(Modal)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
}));

const SettingsContent = styled(Paper)(({ theme }) => ({
  backgroundColor: 'rgba(0, 0, 0, 0.8)',
  boxShadow: theme.shadows[5],
  padding: theme.spacing(4),
  outline: 'none',
  borderRadius: 15,
  maxWidth: 600,
  width: '90%',
}));

const FloatingSettingsButton = styled(Fab)(({ theme }) => ({
  position: 'fixed',
  bottom: theme.spacing(2),
  left: theme.spacing(2),
  background: 'rgba(255, 255, 255, 0.1)',
  backdropFilter: 'blur(5px)',
  '&:hover': {
    background: 'rgba(255, 255, 255, 0.2)',
  },
}));

function AnalysisPanel() {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [config, setConfig] = useState({
    pre_directory_location: '',
    post_directory_location: '',
    nuclei_model_location: '',
    cyto_model_location: '',
    experiment_name: ''
  });
  const [scriptOutput, setScriptOutput] = useState('');
  const [graphPaths, setGraphPaths] = useState([]);
  const [currentGraphIndex, setCurrentGraphIndex] = useState(0);

  useEffect(() => {
    loadConfig();
  }, []);

  useEffect(() => {
    const handlePythonOutput = (event, data) => {
      setScriptOutput(prev => prev + data);
    };
    ipcRenderer.on('python-output', handlePythonOutput);
    return () => {
      ipcRenderer.removeListener('python-output', handlePythonOutput);
    };
  }, []);

  const loadConfig = async () => {
    const loadedConfig = await ipcRenderer.invoke('load-config');
    if (loadedConfig) {
      setConfig(loadedConfig);
    }
  };

  const saveConfig = async () => {
    await ipcRenderer.invoke('save-config', config);
    handleCloseSettings();
  };

  const runAnalysis = async () => {
    setIsLoading(true);
    setScriptOutput('');
    setGraphPaths([]);
    try {
      await ipcRenderer.invoke('save-config', config);
      const pythonResult = await ipcRenderer.invoke('run-python-script', 'AstrocyteAnalysis.py');
      setResult(pythonResult);
      const graphs = await ipcRenderer.invoke('get-graph-paths');
      setGraphPaths(graphs);
    } catch (error) {
      console.error('Error running Python script:', error);
      setResult('Error: ' + error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleOpenSettings = () => setSettingsOpen(true);
  const handleCloseSettings = () => setSettingsOpen(false);

  const browseFolder = async (key) => {
    const result = await ipcRenderer.invoke('open-folder-dialog');
    if (result) {
      setConfig({ ...config, [key]: result });
    }
  };

  const browseFile = async (key) => {
    const result = await ipcRenderer.invoke('open-file-dialog');
    if (result) {
      setConfig({ ...config, [key]: result });
    }
  };

  return (
    <Box sx={{ flexGrow: 1, p: 3, mt: 4 }}>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <GlowElement>
            <StyledPaper>
              <Typography variant="h5" gutterBottom>Controls</Typography>
              <GlowElement sx={{ display: 'inline-block', m: 0.5 }}>
                <StyledButton 
                  variant="contained" 
                  startIcon={isLoading ? <CircularProgress size={24} /> : <PlayArrowIcon />}
                  onClick={runAnalysis}
                  disabled={isLoading}
                >
                  {isLoading ? 'Running...' : 'Start Analysis'}
                </StyledButton>
              </GlowElement>
              {isLoading && (
                <Box sx={{ mt: 2, maxHeight: '200px', overflowY: 'auto', backgroundColor: 'rgba(0,0,0,0.1)', padding: 2, borderRadius: 1 }}>
                  <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{scriptOutput}</pre>
                </Box>
              )}
            </StyledPaper>
          </GlowElement>
        </Grid>
        {result && (
          <Grid item xs={12}>
            <GlowElement>
              <StyledPaper>
                <Typography variant="h5" gutterBottom>Results</Typography>
                <Box sx={{ maxHeight: '300px', overflowY: 'auto' }}>
                  <pre>{JSON.stringify(result, null, 2)}</pre>
                </Box>
              </StyledPaper>
            </GlowElement>
          </Grid>
        )}
        {graphPaths.length > 0 && (
          <Grid item xs={12}>
            <GlowElement>
              <StyledPaper>
                <Typography variant="h5" gutterBottom>Graphs</Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <img 
                    src={graphPaths[currentGraphIndex]} 
                    alt={`Graph ${currentGraphIndex}`} 
                    style={{maxWidth: '100%', maxHeight: '400px', objectFit: 'contain'}} 
                  />
                  <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center', gap: 2 }}>
                    <Button 
                      onClick={() => setCurrentGraphIndex(prev => (prev > 0 ? prev - 1 : graphPaths.length - 1))}
                      disabled={graphPaths.length <= 1}
                    >
                      Previous
                    </Button>
                    <Typography>
                      Graph {currentGraphIndex + 1} of {graphPaths.length}
                    </Typography>
                    <Button 
                      onClick={() => setCurrentGraphIndex(prev => (prev < graphPaths.length - 1 ? prev + 1 : 0))}
                      disabled={graphPaths.length <= 1}
                    >
                      Next
                    </Button>
                  </Box>
                </Box>
              </StyledPaper>
            </GlowElement>
          </Grid>
        )}
      </Grid>

      <GlowElement sx={{ position: 'fixed', bottom: 16, left: 16 }}>
        <FloatingSettingsButton onClick={handleOpenSettings} aria-label="settings">
          <SettingsIcon />
        </FloatingSettingsButton>
      </GlowElement>

      <SettingsModal open={settingsOpen} onClose={handleCloseSettings}>
        <SettingsContent>
          <Typography variant="h5" gutterBottom>Settings</Typography>
          <Grid container spacing={2}>
            {Object.entries(config).map(([key, value]) => (
              <Grid item xs={12} key={key}>
                <GlowElement>
                  <TextField
                    fullWidth
                    label={key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    value={value}
                    onChange={(e) => setConfig({ ...config, [key]: e.target.value })}
                    InputProps={{
                      endAdornment: (
                        <Button onClick={() => key.includes('location') ? 
                          (key.includes('model') ? browseFile(key) : browseFolder(key)) : 
                          null}>
                          <FolderOpenIcon />
                        </Button>
                      ),
                    }}
                    sx={{ '& .MuiInputLabel-root': { whiteSpace: 'normal' } }}
                  />
                </GlowElement>
              </Grid>
            ))}
          </Grid>
          <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
            <Button onClick={handleCloseSettings}>Cancel</Button>
            <Button onClick={saveConfig} variant="contained" sx={{ ml: 1 }}>Save</Button>
          </Box>
        </SettingsContent>
      </SettingsModal>
    </Box>
  );
}

export default AnalysisPanel;