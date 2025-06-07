const API_CONFIG = {
  // API 基础配置
  baseURL: process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000',
  prefix: '/api',
  timeout: 30000, // 30秒超时
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },

  // 重试配置
  retry: {
    maxRetries: 3,
    retryDelay: 1000, // 1秒
    retryableStatusCodes: [408, 429, 500, 502, 503, 504],
  },

  // 模型配置
  models: {
    defaultLLM: 'glm-4-plus',
    defaultEmbedding: 'embedding-3',
  },

  // API 端点
  endpoints: {
    ner: '/api/ner',
    std: '/api/std',
    abbr: '/api/abbr',
    corr: '/api/corr',
    gen: '/api/gen',
  },

  // 错误消息
  errorMessages: {
    networkError: '网络连接失败，请检查网络设置',
    timeout: '请求超时，请稍后重试',
    serverError: '服务器错误，请稍后重试',
    unknownError: '发生未知错误，请稍后重试',
  },
};

export default API_CONFIG; 