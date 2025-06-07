import React from 'react';
import { ChakraProvider, Box } from '@chakra-ui/react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import NERPage from './pages/NERPage';
import StdPage from './pages/StdPage';
import AbbrPage from './pages/AbbrPage';
import CorrPage from './pages/CorrPage';
import GenPage from './pages/GenPage';
import theme from './theme';
import { AppProvider } from './context/AppContext';

// 引入新组件
import ModelOptionsPanel from './components/panels/ModelOptionsPanel';
import ResultPanel from './components/panels/ResultPanel';
import InfoAlert from './components/alerts/InfoAlert';

// 导出组件供页面使用
export { ModelOptionsPanel, ResultPanel, InfoAlert };

// 404页面组件
const NotFound = () => (
  <Box textAlign="center" py={10}>
    <InfoAlert
      title="页面未找到"
      description="抱歉，您访问的页面不存在。"
      type="error"
    />
  </Box>
);

function App() {
  return (
    <ChakraProvider theme={theme}>
      <AppProvider>
        <Router>
          <Box minH="100vh" bg="gray.50">
            <Sidebar />
            <Box ml={{ base: 0, md: 60 }} p="4">
              <Routes>
                {/* 重定向根路径到NER页面 */}
                <Route path="/" element={<Navigate to="/ner" replace />} />
                
                {/* 主要路由 */}
                <Route path="/ner" element={<NERPage />} />
                <Route path="/std" element={<StdPage />} />
                <Route path="/abbr" element={<AbbrPage />} />
                <Route path="/corr" element={<CorrPage />} />
                <Route path="/gen" element={<GenPage />} />
                
                {/* 404页面 */}
                <Route path="*" element={<NotFound />} />
              </Routes>
            </Box>
          </Box>
        </Router>
      </AppProvider>
    </ChakraProvider>
  );
}

export default App;