import React from 'react';
import {
  Center,
  Spinner,
  Text,
  VStack
} from '@chakra-ui/react';

const LoadingSpinner = ({ 
  message = '加载中...',
  size = 'xl',
  thickness = '4px',
  speed = '0.65s',
  color = 'blue.500'
}) => {
  return (
    <Center py={8}>
      <VStack spacing={4}>
        <Spinner
          size={size}
          thickness={thickness}
          speed={speed}
          color={color}
        />
        {message && (
          <Text color="gray.600" fontSize="sm">
            {message}
          </Text>
        )}
      </VStack>
    </Center>
  );
};

export default LoadingSpinner; 