import React from 'react';
import { Alert, AlertIcon, AlertTitle, AlertDescription, Box } from '@chakra-ui/react';

const InfoAlert = ({ 
  title = '使用说明',
  description,
  icon = true,
  variant = 'subtle'
}) => {
  return (
    <Alert 
      status="info" 
      variant={variant}
      flexDirection="column"
      alignItems="center"
      justifyContent="center"
      textAlign="center"
      height="auto"
      p={4}
    >
      {icon && <AlertIcon boxSize="40px" mr={0} />}
      <AlertTitle mt={4} mb={1} fontSize="lg">
        {title}
      </AlertTitle>
      <AlertDescription maxWidth="sm">
        {description}
      </AlertDescription>
    </Alert>
  );
};

export default InfoAlert; 