import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Box,
  VStack,
  Icon,
  Text,
  Flex,
  useColorModeValue,
  Divider,
} from '@chakra-ui/react';
import {
  FiHome,
  FiTag,
  FiFileText,
  FiType,
  FiCheckSquare,
  FiFile,
} from 'react-icons/fi';

const NavItem = ({ icon, children, to, isActive }) => {
  const activeBg = useColorModeValue('brand.50', 'brand.900');
  const activeColor = useColorModeValue('brand.500', 'brand.200');
  const hoverBg = useColorModeValue('gray.100', 'gray.700');
  const navigate = useNavigate();

  return (
    <Flex
      align="center"
      p="4"
      mx="4"
      borderRadius="lg"
      role="group"
      cursor="pointer"
      bg={isActive ? activeBg : 'transparent'}
      color={isActive ? activeColor : 'gray.600'}
      _hover={{
        bg: isActive ? activeBg : hoverBg,
        color: isActive ? activeColor : 'gray.800',
      }}
      onClick={() => navigate(to)}
    >
      <Icon
        mr="4"
        fontSize="16"
        as={icon}
      />
      <Text fontWeight={isActive ? 'semibold' : 'normal'}>
        {children}
      </Text>
    </Flex>
  );
};

const Sidebar = () => {
  const location = useLocation();
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');

  return (
    <Box
      w={{ base: 'full', md: 60 }}
      pos="fixed"
      h="full"
      bg={bgColor}
      borderRight="1px"
      borderColor={borderColor}
    >
      <VStack h="full" py={5} spacing={0}>
        <Box px={4} mb={5}>
          <Text fontSize="2xl" fontWeight="bold" color="brand.500">
            金融文本处理
          </Text>
        </Box>
        <Divider />
        <VStack flex={1} w="full" spacing={1} align="stretch">
          <NavItem
            icon={FiHome}
            to="/"
            isActive={location.pathname === '/'}
          >
            首页
          </NavItem>
          <NavItem
            icon={FiTag}
            to="/ner"
            isActive={location.pathname === '/ner'}
          >
            实体识别
          </NavItem>
          <NavItem
            icon={FiFileText}
            to="/std"
            isActive={location.pathname === '/std'}
          >
            术语标准化
          </NavItem>
          <NavItem
            icon={FiType}
            to="/abbr"
            isActive={location.pathname === '/abbr'}
          >
            缩写展开
          </NavItem>
          <NavItem
            icon={FiCheckSquare}
            to="/corr"
            isActive={location.pathname === '/corr'}
          >
            文本纠错
          </NavItem>
          <NavItem
            icon={FiFile}
            to="/gen"
            isActive={location.pathname === '/gen'}
          >
            文本生成
          </NavItem>
        </VStack>
      </VStack>
    </Box>
  );
};

export default Sidebar;