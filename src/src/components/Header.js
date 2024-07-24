import React, { useState } from 'react';
import { IconButton, Menu, MenuItem, Typography, Box } from '@mui/material';
import SettingsIcon from '@mui/icons-material/Settings';
import { styled, keyframes } from '@mui/system';

const glowingOutline = keyframes`
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
`;

const SettingsButton = styled(IconButton)(({ theme }) => ({
  position: 'relative',
  borderRadius: '15px',
  background: 'rgba(60, 60, 60, 0.6)',
  backdropFilter: 'blur(5px)',
  transition: 'all 0.3s ease',
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
  '&:hover': {
    background: 'rgba(80, 80, 80, 0.7)',
    '&::before': {
      opacity: 1,
    },
  },
}));

function Header() {
  const [anchorEl, setAnchorEl] = useState(null);

  const handleMenu = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  return (
    <Box sx={{ position: 'relative', width: '100%', height: '60px', padding: '10px' }}>
      <SettingsButton
        size="large"
        color="inherit"
        aria-label="menu"
        onClick={handleMenu}
      >
        <SettingsIcon />
      </SettingsButton>
      <Menu
        id="menu-appbar"
        anchorEl={anchorEl}
        anchorOrigin={{
          vertical: 'top',
          horizontal: 'right',
        }}
        keepMounted
        transformOrigin={{
          vertical: 'top',
          horizontal: 'right',
        }}
        open={Boolean(anchorEl)}
        onClose={handleClose}
      >
        <MenuItem onClick={handleClose}>Change Folder</MenuItem>
        <MenuItem onClick={handleClose}>Set Experiment Name</MenuItem>
        <MenuItem onClick={handleClose}>Advanced Settings</MenuItem>
      </Menu>
      <Typography variant="h4" component="div" sx={{ 
        position: 'absolute', 
        top: '15px', 
        left: '80px',
        fontFamily: "'Montserrat', sans-serif",
        fontWeight: 300,
      }}>
        Analysis test
      </Typography>
    </Box>
  );
}

export default Header;