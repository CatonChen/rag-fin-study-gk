import React from 'react';
import { VStack, FormControl, FormLabel, Input, Text } from '@chakra-ui/react';
import { useApp } from '../../context/AppContext';

const ModelOptionsPanel = ({ 
  title = '模型选项',
  description = '配置模型和数据库选项'
}) => {
  const { modelOptions, setModelOptions } = useApp();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setModelOptions({ [name]: value });
  };

  return (
    <VStack spacing={4} align="stretch">
      <Text fontSize="lg" fontWeight="medium">{title}</Text>
      <Text fontSize="sm" color="gray.600">{description}</Text>
      
      <FormControl>
        <FormLabel>大语言模型</FormLabel>
        <Input
          name="model"
          value={modelOptions.model}
          onChange={handleChange}
          placeholder="输入模型名称"
        />
        <Text fontSize="sm" color="gray.500" mt={1}>默认使用智谱AI GLM-4模型</Text>
      </FormControl>

      <FormControl>
        <FormLabel>嵌入模型</FormLabel>
        <Input
          name="embeddingModel"
          value={modelOptions.embeddingModel}
          onChange={handleChange}
          placeholder="输入嵌入模型名称"
        />
        <Text fontSize="sm" color="gray.500" mt={1}>默认使用智谱AI Embedding-3模型</Text>
      </FormControl>

      <FormControl>
        <FormLabel>数据库名称</FormLabel>
        <Input
          name="dbName"
          value={modelOptions.dbName}
          onChange={handleChange}
          placeholder="输入数据库名称"
        />
      </FormControl>
    </VStack>
  );
};

export default ModelOptionsPanel; 