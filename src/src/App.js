import React from 'react';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import AnalysisPanel from './components/AnalysisPanel';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#f48fb1',
    },
  },
  typography: {
    fontFamily: "'Montserrat', sans-serif",
    h4: {
      fontWeight: 300,
    },
    h5: {
      fontWeight: 300,
    },
    button: {
      fontWeight: 500,
    },
    body1: {
      fontWeight: 400,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundColor: 'rgba(66, 66, 66, 0.7)',
          backdropFilter: 'blur(5px)',
        },
      },
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <div className="App" style={{ backgroundColor: '#121212', minHeight: '100vh' }}>
        <AnalysisPanel />
      </div>
    </ThemeProvider>
  );
}

export default App;