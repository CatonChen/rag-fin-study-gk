import React from 'react';
import {
  Card,
  CardBody,
  Heading,
  Button,
  VStack,
  FormControl,
  FormLabel,
  Textarea,
} from '@chakra-ui/react';

const TextInputForm = ({
  title,
  value,
  onChange,
  placeholder,
  onSubmit,
  isLoading,
  submitText = '提交',
  rows = 4,
  children,
}) => {
  return (
    <Card>
      <CardBody>
        <VStack spacing={4} align="stretch">
          <Heading size="md">{title}</Heading>
          
          <FormControl>
            <FormLabel>输入文本</FormLabel>
            <Textarea
              value={value}
              onChange={onChange}
              placeholder={placeholder}
              rows={rows}
            />
          </FormControl>

          {children}

          <Button
            colorScheme="primary"
            size="lg"
            onClick={onSubmit}
            isLoading={isLoading}
            loadingText="处理中..."
            width="full"
          >
            {submitText}
          </Button>
        </VStack>
      </CardBody>
    </Card>
  );
};

export default TextInputForm; 