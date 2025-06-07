import client from './client';
import API_CONFIG from '../config/api';

// 金融实体识别服务
export const nerService = {
  // 识别文本中的金融实体
  recognize: (text, options) => 
    client.post(API_CONFIG.endpoints.ner, { text, options }),
};

// 金融术语标准化服务
export const stdService = {
  // 标准化金融术语
  standardize: (text, options) => 
    client.post(API_CONFIG.endpoints.std, { text, options }),
};

// 金融术语缩写展开服务
export const abbrService = {
  // 展开金融术语缩写
  expand: (text, options) => 
    client.post(API_CONFIG.endpoints.abbr, { text, options }),
};

// 金融文本纠错服务
export const corrService = {
  // 纠正金融文本
  correct: (text, options) => 
    client.post(API_CONFIG.endpoints.corr, { text, options }),
};

// 金融文本生成服务
export const genService = {
  // 生成金融文本
  generate: (text, options) => 
    client.post(API_CONFIG.endpoints.gen, { text, options }),
}; 