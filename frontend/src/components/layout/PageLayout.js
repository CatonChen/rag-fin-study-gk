import React from 'react';
import {
  Container,
  Box,
  Heading,
  Text,
  VStack,
} from '@chakra-ui/react';
import ErrorAlert from '../ErrorAlert';
import LoadingSpinner from '../LoadingSpinner';

const PageLayout = ({
  title,
  description,
  isLoading,
  loadingMessage,
  error,
  onErrorClose,
  children,
}) => {
  return (
    <Container maxW="container.xl" py={8}>
      <VStack spacing={8} align="stretch">
        {/* 页面标题和描述 */}
        <Box>
          <Heading as="h1" size="xl" mb={2} color="primary.500">
            {title}
          </Heading>
          <Text color="gray.600">
            {description}
          </Text>
        </Box>

        {/* 错误提示 */}
        <ErrorAlert 
          message={error}
          onClose={onErrorClose}
        />

        {/* 加载状态 */}
        {isLoading ? (
          <LoadingSpinner message={loadingMessage} />
        ) : (
          children
        )}
      </VStack>
    </Container>
  );
};

export default PageLayout; 