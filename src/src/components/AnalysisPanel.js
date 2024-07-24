import React, { useState } from 'react';
import { Box, Button, Typography, Paper, Grid } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
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

function AnalysisPanel() {
  const [result, setResult] = useState(null);

  const runAnalysis = async () => {
    try {
      const pythonResult = await ipcRenderer.invoke('run-python-script', 'sample_analysis.py', ['arg1', 'arg2']);
      setResult(pythonResult);
    } catch (error) {
      console.error('Error running Python script:', error);
      setResult('Error: ' + error.message);
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
                <StyledButton variant="contained" startIcon={<PlayArrowIcon />} onClick={runAnalysis}>
                  Start
                </StyledButton>
              </GlowElement>
              <GlowElement sx={{ display: 'inline-block', m: 0.5 }}>
                <StyledButton variant="contained" startIcon={<StopIcon />} color="secondary">
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
              <GlowElement sx={{ display: 'inline-block', m: 0.5 }}>
                <StyledButton variant="outlined">Test</StyledButton>
              </GlowElement>
              <GlowElement sx={{ display: 'inline-block', m: 0.5 }}>
                <StyledButton variant="outlined">Test</StyledButton>
              </GlowElement>
              <GlowElement sx={{ display: 'inline-block', m: 0.5 }}>
                <StyledButton variant="outlined">Test</StyledButton>
              </GlowElement>
              <GlowElement sx={{ display: 'inline-block', m: 0.5 }}>
                <StyledButton variant="outlined">Test</StyledButton>
              </GlowElement>
            </StyledPaper>
          </GlowElement>
        </Grid>
        {result && (
          <Grid item xs={12}>
            <GlowElement>
              <StyledPaper>
                <Typography variant="h5" gutterBottom>Results</Typography>
                <pre>{JSON.stringify(result, null, 2)}</pre>
              </StyledPaper>
            </GlowElement>
          </Grid>
        )}
      </Grid>
    </Box>
  );
}

export default AnalysisPanel;