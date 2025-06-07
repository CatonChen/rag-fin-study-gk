const ENV_CONFIG = {
  // 环境变量
  NODE_ENV: process.env.NODE_ENV || 'development',
  
  // API配置
  API_BASE_URL: process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000',
  
  // 应用配置
  APP_NAME: '金融文本处理系统',
  APP_VERSION: '1.0.0',
  
  // 功能开关
  FEATURES: {
    ENABLE_DARK_MODE: true,
    ENABLE_ERROR_BOUNDARY: true,
    ENABLE_ANALYTICS: false,
  },
  
  // 性能配置
  PERFORMANCE: {
    API_TIMEOUT: 30000,
    API_RETRY_COUNT: 3,
    API_RETRY_DELAY: 1000,
  },
  
  // 缓存配置
  CACHE: {
    ENABLED: true,
    TTL: 3600, // 1小时
  },
  
  // 日志配置
  LOGGING: {
    ENABLED: process.env.NODE_ENV === 'development',
    LEVEL: process.env.NODE_ENV === 'development' ? 'debug' : 'error',
  },
};

export default ENV_CONFIG; 