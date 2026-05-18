"""
test_cag_qwen_final.py - CAG + Qwen2.5 最終版
=============================================
改進版：更長超時 + 明確上下文 + 重試機制
"""

import os
import json
import requests
from datetime import datetime
import time


class QwenMusicTester:
    """Qwen 音樂樂理測試器"""
    
    # 明確的音樂理論上下文
    MUSIC_CONTEXT = """
你是一個音樂理論專家。當回答音樂相關問題時，請專注於音樂理論、樂理、作曲技巧等音樂領域。
如果被問到「模式」或「Mode」等術語，請在音樂上下文中解釋（如大調模式、小調模式、教會模式等），
而不是統計學中的眾數概念。
"""
    
    def __init__(self, 
                 model="qwen2.5:7b",
                 ollama_url="http://localhost:11434/api/generate",
                 timeout=60):  # 增加到 60 秒
        self.model = model
        self.ollama_url = ollama_url
        self.timeout = timeout
        self.results = []
    
    def test_connection(self, retries=3):
        """測試連接 (帶重試)"""
        print("\n" + "="*70)
        print("🔗 檢查本地模型連接")
        print("="*70)
        
        for attempt in range(1, retries + 1):
            try:
                print(f"\n嘗試 {attempt}/{retries}...")
                response = requests.get(
                    self.ollama_url.replace("/api/generate", "/api/tags"),
                    timeout=5
                )
                
                if response.status_code == 200:
                    print(f"✓ 成功連接到 Ollama")
                    data = response.json()
                    models = data.get("models", [])
                    print(f"✓ 可用模型: {len(models)} 個")
                    return True
            except requests.exceptions.Timeout:
                if attempt < retries:
                    print(f"  ⏳ 連接超時，{2 ** attempt} 秒後重試...")
                    time.sleep(2 ** attempt)
            except Exception as e:
                if attempt < retries:
                    print(f"  ⚠️  {str(e)[:50]}，重試中...")
                    time.sleep(1)
        
        print(f"\n✗ 無法連接到 Ollama (嘗試 {retries} 次)")
        return False
    
    def ask_question(self, question, retries=2):
        """提問 (帶重試機制)"""
        
        # 構建清晰的提示詞
        prompt = f"""{self.MUSIC_CONTEXT}

問題: {question}

請用中文回答，限制在 300 字以內。"""
        
        for attempt in range(1, retries + 1):
            try:
                print(f"  ⏳ 發送請求... (嘗試 {attempt}/{retries})", end="", flush=True)
                
                response = requests.post(
                    self.ollama_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0.7
                    },
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("response", "").strip()
                    print(f" ✓")
                    return answer
                else:
                    print(f" ✗ (狀態碼: {response.status_code})")
                    
            except requests.exceptions.Timeout:
                print(f" ✗ (超時)")
                if attempt < retries:
                    print(f"  ⏳ {2 ** attempt} 秒後重試...")
                    time.sleep(2 ** attempt)
            except Exception as e:
                print(f" ✗")
                if attempt < retries:
                    print(f"  ⚠️  錯誤: {str(e)[:40]}")
                    time.sleep(1)
        
        return "[失敗: 無法從 Ollama 獲得回答]"
    
    def run_music_theory_tests(self):
        """運行樂理測試"""
        print("\n" + "="*70)
        print("🎵 樂理問題測試")
        print("="*70)
        
        questions = [
            "什麼是對位法？它有什麼特點？",
            "解釋音樂中的和聲進行。",
            "什麼是奏鳴曲式？主要由哪些部分組成？",
            "解釋弦樂四重奏的樂器配置。",
            "在音樂理論中，什麼是模式 (Mode)？舉例說明大調模式、小調模式等。",
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n【問題 {i}/{len(questions)}】{question}")
            print("-" * 70)
            
            # 第一個問題前增加預熱時間
            if i == 1:
                print("⏳ 模型預熱中... (首次加載可能需要較長時間)")
                time.sleep(2)
            
            answer = self.ask_question(question)
            
            # 顯示回答
            if len(answer) > 300:
                print(f"✓ 回答:\n{answer[:300]}...\n")
            else:
                print(f"✓ 回答:\n{answer}\n")
            
            self.results.append({
                "題號": i,
                "問題": question,
                "回答": answer,
                "時間": datetime.now().isoformat()
            })
    
    def save_results(self):
        """保存測試結果"""
        output = {
            "測試時間": datetime.now().isoformat(),
            "模型": self.model,
            "API": "Ollama",
            "超時設定": f"{self.timeout} 秒",
            "結果": self.results,
            "成功率": f"{len([r for r in self.results if not r['回答'].startswith('[失敗')])} / {len(self.results)}"
        }
        
        output_file = "cag_qwen_test_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 結果已保存至: {output_file}")
        return output_file


def main():
    """主程序"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  🎵 CAG + Qwen2.5 最終版測試 🎵".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    # 初始化測試器
    tester = QwenMusicTester(timeout=60)  # 60 秒超時
    
    # 檢查連接
    if not tester.test_connection():
        print("\n提示:")
        print("  1. 確保 Ollama 已安裝並運行")
        print("  2. 執行: ollama serve")
        print("  3. 拉取模型: ollama pull qwen2.5:7b")
        return
    
    # 運行測試
    tester.run_music_theory_tests()
    
    # 保存結果
    tester.save_results()
    
    # 顯示摘要
    print("\n" + "="*70)
    print("📊 測試完成！")
    print("="*70)
    print(f"✓ 總題數: {len(tester.results)}")
    print(f"✓ 成功: {len([r for r in tester.results if not r['回答'].startswith('[失敗')])}")
    print(f"✓ 結果文件: cag_qwen_test_results.json")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
