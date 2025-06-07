import axios from 'axios';
import API_CONFIG from '../config/api';

// 创建axios实例
const client = axios.create({
    baseURL: API_CONFIG.baseURL,
    timeout: API_CONFIG.timeout,
    headers: API_CONFIG.headers
});

// 请求拦截器
client.interceptors.request.use(
  (config) => {
    // 添加默认的模型参数
    if (config.data) {
      config.data = {
        ...config.data,
        model_name: config.data.model_name || API_CONFIG.models.defaultLLM,
        temperature: config.data.temperature || 0.7,
        top_p: config.data.top_p || 0.9,
        max_tokens: config.data.max_tokens || 2000
      };
    }

    // 记录请求日志
    console.log(`[Request] ${config.method.toUpperCase()} ${config.url}`, config.data);

    return config;
  },
  (error) => {
    console.error('[Request Error]', error);
    return Promise.reject(error);
  }
);

// 响应拦截器
client.interceptors.response.use(
  (response) => {
    // 记录响应日志
    console.log(`[Response] ${response.config.method.toUpperCase()} ${response.config.url}`, response.data);

    // 处理响应数据
    if (response.data && response.data.error) {
      console.error('[Response Error]', response.data.error);
      return Promise.reject(new Error(response.data.error));
    }

    return response.data;
  },
  (error) => {
    // 对响应错误做点什么
    if (error.response) {
      // 服务器返回错误状态码
      const errorMessage = (() => {
        switch (error.response.status) {
          case 400:
            return API_CONFIG.errorMessages.badRequest || '请求参数错误，请检查输入';
          case 401:
            return API_CONFIG.errorMessages.unauthorized || '未授权，请重新登录';
          case 403:
            return API_CONFIG.errorMessages.forbidden || '拒绝访问，请检查权限';
          case 404:
            return API_CONFIG.errorMessages.notFound || '请求的资源不存在';
          case 500:
            return API_CONFIG.errorMessages.serverError || '服务器内部错误，请稍后重试';
          default:
            return API_CONFIG.errorMessages.unknownError || '未知错误，请稍后重试';
        }
      })();

      console.error(`[Response Error] ${error.response.status}: ${errorMessage}`);
      return Promise.reject(new Error(errorMessage));
    } else if (error.request) {
      // 请求发出但没有收到响应
      console.error('[Network Error]', API_CONFIG.errorMessages.networkError);
      return Promise.reject(new Error(API_CONFIG.errorMessages.networkError));
    } else {
      // 请求配置出错
      console.error('[Request Error]', error.message);
      return Promise.reject(error);
    }
  }
);

export default client; 