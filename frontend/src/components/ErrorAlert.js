import React from 'react';
import {
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  CloseButton,
  Box
} from '@chakra-ui/react';

const ErrorAlert = ({ 
  title = '错误', 
  message, 
  onClose,
  status = 'error'
}) => {
  if (!message) return null;

  return (
    <Alert 
      status={status}
      variant="subtle"
      flexDirection="column"
      alignItems="center"
      justifyContent="center"
      textAlign="center"
      borderRadius="md"
      mb={4}
    >
      <Box display="flex" width="100%" alignItems="center">
        <AlertIcon />
        <AlertTitle mr={2}>{title}</AlertTitle>
        {onClose && (
          <CloseButton 
            position="absolute" 
            right="8px" 
            top="8px" 
            onClick={onClose}
          />
        )}
      </Box>
      <AlertDescription mt={2}>
        {message}
      </AlertDescription>
    </Alert>
  );
};

export default ErrorAlert; 