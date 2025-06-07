import React from 'react';

// LLM选项组件
export const LLMOptions = ({ options, onChange }) => {
  return (
    <div className="mb-4">
      <h3 className="text-lg font-medium mb-2">大语言模型设置</h3>
      <div className="grid grid-cols-1 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700">模型</label>
          <input
            type="text"
            name="model"
            value={options.model || 'glm-4-plus'}
            onChange={onChange}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
            disabled
          />
          <p className="text-sm text-gray-500 mt-1">使用智谱AI GLM-4模型</p>
        </div>
      </div>
    </div>
  );
};

// 向量数据库选项组件
export const EmbeddingOptions = ({ options, onChange }) => {
  return (
    <div className="grid grid-cols-2 gap-4 mb-4">
      <div>
        <label className="block text-sm font-medium text-gray-700">嵌入模型</label>
        <input
          type="text"
          name="model"
          value={options.model || 'embedding-3'}
          onChange={onChange}
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
          disabled
        />
        <p className="text-sm text-gray-500 mt-1">使用智谱AI Embedding-3模型</p>
      </div>
      
      <div>
        <label className="block text-sm font-medium text-gray-700">数据库名称</label>
        <input
          type="text"
          name="dbName"
          value={options.dbName || 'financial_terms_zhipu'}
          onChange={onChange}
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
          disabled
        />
      </div>
    </div>
  );
};

// 通用输入文本区域组件
export const TextInput = ({ value, onChange, rows = 6, placeholder }) => {
  return (
    <textarea
      className="w-full p-2 border rounded-md mb-4"
      rows={rows}
      placeholder={placeholder}
      value={value}
      onChange={onChange}
    />
  );
}; 