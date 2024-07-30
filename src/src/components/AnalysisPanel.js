import React, { useState, useEffect } from 'react';
import { Box, Button, Typography, Paper, Grid, Modal, TextField, Fab } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';
import SettingsIcon from '@mui/icons-material/Settings';
import { styled, keyframes } from '@mui/system';

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
  const [settings, setSettings] = useState({
    preDirectory: '',
    postDirectory: '',
    nucleiModel: '',
    cytoModel: ''
  });

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    const loadedSettings = await ipcRenderer.invoke('load-settings');
    if (loadedSettings) {
      setSettings(loadedSettings);
    }
  };

  const saveSettings = async () => {
    await ipcRenderer.invoke('save-settings', settings);
    handleCloseSettings();
  };

  const runAnalysis = async () => {
    setIsLoading(true);
    try {
      const pythonResult = await ipcRenderer.invoke('run-python-script', 'sample_analysis.py', ['arg1', 'arg2']);
      setResult(pythonResult);
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
      setSettings({ ...settings, [key]: result });
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
                  startIcon={<PlayArrowIcon />} 
                  onClick={runAnalysis}
                  disabled={isLoading}
                >
                  {isLoading ? 'Running...' : 'Start'}
                </StyledButton>
              </GlowElement>
              <GlowElement sx={{ display: 'inline-block', m: 0.5 }}>
                <StyledButton 
                  variant="contained" 
                  startIcon={<StopIcon />} 
                  color="secondary"
                  disabled={!isLoading}
                >
                  Stop
                </StyledButton>
              </GlowElement>
            </StyledPaper>
          </GlowElement>
        </Grid>
        <Grid item xs={12}>
          <GlowElement>
            <StyledPaper>
              <Typography variant="h5" gutterBottom>Analysis Options</Typography>
              {['Option 1', 'Option 2', 'Option 3', 'Option 4'].map((option, index) => (
                <GlowElement key={index} sx={{ display: 'inline-block', m: 0.5 }}>
                  <StyledButton variant="outlined">{option}</StyledButton>
                </GlowElement>
              ))}
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
            {Object.entries(settings).map(([key, value]) => (
              <Grid item xs={12} key={key}>
                <GlowElement>
                  <TextField
                    fullWidth
                    label={key}
                    value={value}
                    onChange={(e) => setSettings({ ...settings, [key]: e.target.value })}
                    InputProps={{
                      endAdornment: (
                        <Button onClick={() => browseFolder(key)}>
                          <FolderOpenIcon />
                        </Button>
                      ),
                    }}
                  />
                </GlowElement>
              </Grid>
            ))}
          </Grid>
          <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
            <Button onClick={handleCloseSettings}>Cancel</Button>
            <Button onClick={saveSettings} variant="contained" sx={{ ml: 1 }}>Save</Button>
          </Box>
        </SettingsContent>
      </SettingsModal>
    </Box>
  );
}

export default AnalysisPanel;