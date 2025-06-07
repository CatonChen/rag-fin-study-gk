import React from 'react';
import { Box, Alert, AlertIcon, AlertTitle, AlertDescription, Text } from '@chakra-ui/react';

const ResultPanel = ({ result, title = '处理结果', type = 'info' }) => {
  if (!result) return null;

  return (
    <Box mt={6}>
      <Alert status={type} variant="left-accent">
        <AlertIcon />
        <Box>
          <AlertTitle>{title}</AlertTitle>
          <AlertDescription>
            <Text whiteSpace="pre-wrap">{result}</Text>
          </AlertDescription>
        </Box>
      </Alert>
    </Box>
  );
};

export default ResultPanel; 