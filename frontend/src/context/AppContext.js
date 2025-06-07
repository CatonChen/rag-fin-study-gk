import React, { createContext, useContext, useReducer } from 'react';

// 初始状态
const initialState = {
  modelOptions: {
    model: 'glm-4-plus',
    embeddingModel: 'embedding-3',
    dbName: 'financial_terms_zhipu'
  },
  loading: false,
  error: null
};

// Action 类型
const ActionTypes = {
  SET_MODEL_OPTIONS: 'SET_MODEL_OPTIONS',
  SET_LOADING: 'SET_LOADING',
  SET_ERROR: 'SET_ERROR',
  CLEAR_ERROR: 'CLEAR_ERROR'
};

// Reducer
const reducer = (state, action) => {
  switch (action.type) {
    case ActionTypes.SET_MODEL_OPTIONS:
      return {
        ...state,
        modelOptions: {
          ...state.modelOptions,
          ...action.payload
        }
      };
    case ActionTypes.SET_LOADING:
      return {
        ...state,
        loading: action.payload
      };
    case ActionTypes.SET_ERROR:
      return {
        ...state,
        error: action.payload
      };
    case ActionTypes.CLEAR_ERROR:
      return {
        ...state,
        error: null
      };
    default:
      return state;
  }
};

// 创建 Context
const AppContext = createContext();

// Provider 组件
export const AppProvider = ({ children }) => {
  const [state, dispatch] = useReducer(reducer, initialState);

  const value = {
    ...state,
    setModelOptions: (options) => {
      dispatch({ type: ActionTypes.SET_MODEL_OPTIONS, payload: options });
    },
    setLoading: (loading) => {
      dispatch({ type: ActionTypes.SET_LOADING, payload: loading });
    },
    setError: (error) => {
      dispatch({ type: ActionTypes.SET_ERROR, payload: error });
    },
    clearError: () => {
      dispatch({ type: ActionTypes.CLEAR_ERROR });
    }
  };

  return (
    <AppContext.Provider value={value}>
      {children}
    </AppContext.Provider>
  );
};

// 自定义 Hook
export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
}; 