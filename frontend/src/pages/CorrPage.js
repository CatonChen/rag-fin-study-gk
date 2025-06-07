import React, { useState } from 'react';
import { Box, VStack, Button, useToast } from '@chakra-ui/react';
import { ModelOptionsPanel, ResultPanel, InfoAlert } from '../App';
import { correctTerms } from '../api/terms';
import { useApp } from '../context/AppContext';

const CorrPage = () => {
  const [input, setInput] = useState('');
  const [result, setResult] = useState('');
  const { modelOptions, setLoading, setError, clearError } = useApp();
  const toast = useToast();

  const handleSubmit = async () => {
    if (!input.trim()) {
      toast({
        title: '错误',
        description: '请输入需要纠错的术语',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    setLoading(true);
    clearError();
    try {
      const response = await correctTerms(input, modelOptions);
      setResult(response.result);
    } catch (error) {
      setError(error.message || '处理失败');
      toast({
        title: '错误',
        description: error.message || '处理失败',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box maxW="4xl" mx="auto">
      <InfoAlert
        title="术语纠错"
        description="输入可能存在错误的金融术语，系统将自动进行纠错。"
      />

      <VStack spacing={6} mt={6}>
        <ModelOptionsPanel
          title="模型配置"
          description="配置使用的模型和数据库"
        />

        <Box w="100%">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="请输入需要纠错的术语"
            rows={6}
            style={{
              width: '100%',
              padding: '0.5rem',
              border: '1px solid #E2E8F0',
              borderRadius: '0.375rem',
              marginBottom: '1rem'
            }}
          />
          <Button
            colorScheme="blue"
            onClick={handleSubmit}
            isLoading={loading}
            loadingText="处理中"
            width="100%"
          >
            开始处理
          </Button>
        </Box>

        <ResultPanel
          result={result}
          title="纠错结果"
          type="success"
        />
      </VStack>
    </Box>
  );
};

export default CorrPage;