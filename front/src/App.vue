<script setup>
import { ref, computed } from 'vue';
import { marked } from 'marked';

// 配置marked
marked.setOptions({
  breaks: true,
  gfm: true
});

// 响应式数据
const question = ref('');
const kbId = ref('chart-mrag');
const answer = ref('');
const isLoading = ref(false);
const error = ref('');
const history = ref([]);

// API地址
const API_BASE_URL = 'http://localhost:8000';

// 计算属性：将markdown转换为HTML
const renderedAnswer = computed(() => {
  if (!answer.value) return '';
  return marked(answer.value);
});

// 发送查询
const submitQuery = async () => {
  if (!question.value.trim()) {
    error.value = '请输入问题';
    return;
  }

  isLoading.value = true;
  error.value = '';
  answer.value = '';

  try {
    const response = await fetch(`${API_BASE_URL}/api/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        question: question.value,
        kb_id: kbId.value
      })
    });

    const data = await response.json();

    if (data.success) {
      answer.value = data.answer;
      // 添加到历史记录
      history.value.unshift({
        question: question.value,
        answer: data.answer,
        timestamp: new Date().toLocaleString()
      });
      // 限制历史记录数量
      if (history.value.length > 10) {
        history.value.pop();
      }
    } else {
      error.value = data.error || '查询失败';
    }
  } catch (err) {
    error.value = `网络错误: ${err.message}。请确保后端服务已启动 (http://localhost:8000)`;
    console.error('查询错误:', err);
  } finally {
    isLoading.value = false;
  }
};

// 清空当前内容
const clearAll = () => {
  question.value = '';
  answer.value = '';
  error.value = '';
};

// 加载历史记录中的问题
const loadFromHistory = (item) => {
  question.value = item.question;
  answer.value = item.answer;
};

// 按下Enter键提交
const handleKeydown = (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    submitQuery();
  }
};
</script>

<template>
  <div class="app-container">
    <!-- 头部 -->
    <header class="app-header">
      <div class="header-content">
        <img alt="Vue logo" class="logo" src="./assets/logo.svg" width="40" height="40" />
        <h1 class="title">NightStarAgent RAG</h1>
      </div>
      <div class="kb-selector">
        <label for="kbId">知识库:</label>
        <input
          id="kbId"
          v-model="kbId"
          type="text"
          placeholder="chart-mrag"
          class="kb-input"
        />
      </div>
    </header>

    <!-- 主要内容区 -->
    <main class="main-content">
      <!-- 输入区域 -->
      <div class="input-section">
        <div class="input-wrapper">
          <textarea
            v-model="question"
            :disabled="isLoading"
            @keydown="handleKeydown"
            placeholder="请输入您的问题，按 Enter 发送..."
            class="question-input"
            rows="3"
          ></textarea>
          <div class="input-actions">
            <button
              @click="clearAll"
              :disabled="isLoading"
              class="btn btn-secondary"
            >
              清空
            </button>
            <button
              @click="submitQuery"
              :disabled="isLoading || !question.trim()"
              class="btn btn-primary"
            >
              <span v-if="isLoading">
                <span class="loading-spinner"></span>
                处理中...
              </span>
              <span v-else>发送</span>
            </button>
          </div>
        </div>
      </div>

      <!-- 错误提示 -->
      <div v-if="error" class="error-message">
        <span class="error-icon">⚠️</span>
        {{ error }}
      </div>

      <!-- 回答区域 -->
      <div v-if="answer || isLoading" class="answer-section">
        <div class="section-header">
          <h3>回答</h3>
          <span v-if="isLoading" class="loading-text">正在生成回答...</span>
        </div>
        
        <!-- 加载状态 -->
        <div v-if="isLoading" class="loading-container">
          <div class="loading-spinner-large"></div>
          <p class="loading-text">正在思考中，请稍候...</p>
        </div>

        <!-- Markdown 渲染区域 -->
        <div
          v-if="!isLoading && answer"
          class="markdown-content"
          v-html="renderedAnswer"
        ></div>
      </div>

      <!-- 历史记录 -->
      <div v-if="history.length > 0" class="history-section">
        <div class="section-header">
          <h3>历史记录</h3>
        </div>
        <div class="history-list">
          <div
            v-for="(item, index) in history"
            :key="index"
            @click="loadFromHistory(item)"
            class="history-item"
          >
            <div class="history-question">
              <span class="history-icon">Q:</span>
              {{ item.question }}
            </div>
            <div class="history-timestamp">{{ item.timestamp }}</div>
          </div>
        </div>
      </div>
    </main>

    <!-- 底部 -->
    <footer class="app-footer">
      <p>NightStarAgent RAG - 智能问答系统</p>
    </footer>
  </div>
</template>

<style scoped>
/* 全局样式 */
.app-container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* 头部样式 */
.app-header {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  box-shadow: 0 2px 20px rgba(0, 0, 0, 0.1);
  padding: 1rem 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-content {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.logo {
  filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.1));
}

.title {
  font-size: 1.5rem;
  font-weight: 700;
  color: #2c3e50;
  margin: 0;
}

.kb-selector {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.9rem;
  color: #666;
}

.kb-input {
  padding: 0.5rem 1rem;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 0.9rem;
  outline: none;
  transition: border-color 0.3s;
}

.kb-input:focus {
  border-color: #667eea;
}

/* 主要内容区 */
.main-content {
  flex: 1;
  padding: 2rem;
  max-width: 1000px;
  margin: 0 auto;
  width: 100%;
  box-sizing: border-box;
}

/* 输入区域 */
.input-section {
  margin-bottom: 1.5rem;
}

.input-wrapper {
  background: white;
  border-radius: 16px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
  overflow: hidden;
}

.question-input {
  width: 100%;
  padding: 1.5rem;
  border: none;
  outline: none;
  font-size: 1rem;
  line-height: 1.6;
  resize: none;
  background: transparent;
  box-sizing: border-box;
}

.question-input:disabled {
  background: #f8f9fa;
  cursor: not-allowed;
}

.question-input::placeholder {
  color: #adb5bd;
}

.input-actions {
  display: flex;
  justify-content: flex-end;
  padding: 0.75rem 1.5rem 1.5rem;
  gap: 0.75rem;
}

/* 按钮样式 */
.btn {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-primary {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.btn-primary:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}

.btn-secondary {
  background: #f1f3f5;
  color: #495057;
}

.btn-secondary:hover:not(:disabled) {
  background: #e9ecef;
}

/* 加载动画 */
.loading-spinner {
  width: 16px;
  height: 16px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

.loading-spinner-large {
  width: 50px;
  height: 50px;
  border: 3px solid rgba(102, 126, 234, 0.2);
  border-top-color: #667eea;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.loading-text {
  color: #667eea;
  font-weight: 500;
}

.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 3rem;
  gap: 1rem;
}

/* 错误提示 */
.error-message {
  background: #fff5f5;
  border: 1px solid #fc8181;
  border-radius: 12px;
  padding: 1rem 1.5rem;
  margin-bottom: 1.5rem;
  color: #c53030;
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.error-icon {
  font-size: 1.2rem;
}

/* 区域头部 */
.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.section-header h3 {
  font-size: 1.1rem;
  font-weight: 600;
  color: #2c3e50;
  margin: 0;
}

/* 回答区域 */
.answer-section {
  background: white;
  border-radius: 16px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  padding: 1.5rem;
  margin-bottom: 1.5rem;
}

/* Markdown 内容样式 */
.markdown-content {
  line-height: 1.8;
  color: #2c3e50;
}

.markdown-content :deep(h1) {
  font-size: 1.75rem;
  font-weight: 700;
  color: #2c3e50;
  margin-top: 1.5rem;
  margin-bottom: 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 2px solid #667eea;
}

.markdown-content :deep(h2) {
  font-size: 1.5rem;
  font-weight: 600;
  color: #2c3e50;
  margin-top: 1.25rem;
  margin-bottom: 0.75rem;
}

.markdown-content :deep(h3) {
  font-size: 1.25rem;
  font-weight: 600;
  color: #2c3e50;
  margin-top: 1rem;
  margin-bottom: 0.5rem;
}

.markdown-content :deep(h4) {
  font-size: 1.1rem;
  font-weight: 600;
  color: #2c3e50;
  margin-top: 0.75rem;
  margin-bottom: 0.5rem;
}

.markdown-content :deep(h5),
.markdown-content :deep(h6) {
  font-size: 1rem;
  font-weight: 600;
  color: #2c3e50;
  margin-top: 0.5rem;
  margin-bottom: 0.25rem;
}

.markdown-content :deep(p) {
  margin-bottom: 1rem;
}

.markdown-content :deep(strong) {
  font-weight: 600;
  color: #667eea;
}

.markdown-content :deep(em) {
  font-style: italic;
}

.markdown-content :deep(code) {
  background: #f1f3f5;
  padding: 0.125rem 0.375rem;
  border-radius: 4px;
  font-family: 'Fira Code', 'Monaco', monospace;
  font-size: 0.9em;
  color: #e74c3c;
}

.markdown-content :deep(pre) {
  background: #2c3e50;
  padding: 1rem;
  border-radius: 8px;
  overflow-x: auto;
  margin-bottom: 1rem;
}

.markdown-content :deep(pre code) {
  background: transparent;
  color: #ecf0f1;
  padding: 0;
}

/* 列表样式 */
.markdown-content :deep(ul) {
  list-style-type: disc;
  padding-left: 1.5rem;
  margin-bottom: 1rem;
}

.markdown-content :deep(ol) {
  list-style-type: decimal;
  padding-left: 1.5rem;
  margin-bottom: 1rem;
}

.markdown-content :deep(li) {
  margin-bottom: 0.25rem;
}

/* 表格样式 */
.markdown-content :deep(table) {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 1rem;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.markdown-content :deep(thead) {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.markdown-content :deep(th) {
  padding: 0.75rem 1rem;
  text-align: left;
  font-weight: 600;
  color: white;
  border: 1px solid #5a67d8;
}

.markdown-content :deep(td) {
  padding: 0.75rem 1rem;
  border: 1px solid #e2e8f0;
  color: #2c3e50;
}

.markdown-content :deep(tbody tr:nth-child(even)) {
  background: #f8f9fa;
}

.markdown-content :deep(tbody tr:hover) {
  background: #f1f3f5;
}

/* 引用块样式 */
.markdown-content :deep(blockquote) {
  border-left: 4px solid #667eea;
  padding-left: 1rem;
  margin: 1rem 0;
  color: #6c757d;
  background: #f8f9fa;
  padding: 1rem;
  border-radius: 0 8px 8px 0;
}

/* 分隔线样式 */
.markdown-content :deep(hr) {
  border: none;
  border-top: 2px solid #e2e8f0;
  margin: 1.5rem 0;
}

/* 链接样式 */
.markdown-content :deep(a) {
  color: #667eea;
  text-decoration: none;
  border-bottom: 1px solid transparent;
  transition: border-color 0.3s;
}

.markdown-content :deep(a:hover) {
  border-bottom-color: #667eea;
}

/* 历史记录 */
.history-section {
  background: white;
  border-radius: 16px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  padding: 1.5rem;
}

.history-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.history-item {
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.history-item:hover {
  background: #e9ecef;
  transform: translateX(4px);
}

.history-question {
  display: flex;
  align-items: flex-start;
  gap: 0.5rem;
  color: #2c3e50;
  margin-bottom: 0.25rem;
}

.history-icon {
  font-weight: 600;
  color: #667eea;
}

.history-timestamp {
  font-size: 0.75rem;
  color: #adb5bd;
}

/* 底部 */
.app-footer {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  padding: 1rem;
  text-align: center;
  color: #6c757d;
  font-size: 0.875rem;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .main-content {
    padding: 1rem;
  }

  .app-header {
    padding: 0.75rem 1rem;
    flex-direction: column;
    gap: 0.75rem;
  }

  .title {
    font-size: 1.25rem;
  }

  .input-actions {
    flex-direction: column;
  }

  .btn {
    width: 100%;
    justify-content: center;
  }

  .markdown-content :deep(table) {
    font-size: 0.875rem;
  }

  .markdown-content :deep(th),
  .markdown-content :deep(td) {
    padding: 0.5rem;
  }
}
</style>
