# lambda/index.py
import json
import os
import boto3
import re  # 正規表現モジュールをインポート
from botocore.exceptions import ClientError
import urllib.requests
import time


class LLMClient:
    """LLM API クライアントクラス"""
    
    def __init__(self, api_url):
        """
        初期化
        
        Args:
            api_url (str): API のベース URL（ngrok URL）
        """
        self.api_url = api_url.rstrip('/')
        self.session = urllib.requests.Session()
    
    def health_check(self):
        """
        ヘルスチェック
        
        Returns:
            dict: ヘルスチェック結果
        """
        response = self.session.get(f"{self.api_url}/health")
        return response.json()
    
    def generate(self, prompt, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True):
        """
        テキスト生成
        
        Args:
            prompt (str): プロンプト文字列
            max_new_tokens (int, optional): 生成する最大トークン数
            temperature (float, optional): 温度パラメータ
            top_p (float, optional): top-p サンプリングのパラメータ
            do_sample (bool, optional): サンプリングを行うかどうか
        
        Returns:
            dict: 生成結果
        """
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample
        }
        
        start_time = time.time()
        response = self.session.post(
            f"{self.api_url}/generate",
            json=payload
        )
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            result["total_request_time"] = total_time
            return result
        else:
            raise Exception(f"API error: {response.status_code} - {response.text}")

def lambda_handler(event, context):
    try:
        # ngrok URLを設定（実際のURLに置き換えてください）
        NGROK_URL = "https://your-ngrok-url.ngrok.url"
        
        # クライアントの初期化
        client = LLMClient(NGROK_URL)
        
        # リクエストボディの解析
        body = json.loads(event['body'])
        message = body['message']
        conversation_history = body.get('conversationHistory', [])
        
        print("Processing message:", message)
        
        # 会話履歴を使用
        messages = conversation_history.copy()
        
        # ユーザーメッセージを追加
        messages.append({
            "role": "user",
            "content": message
        })
        
        # Nova Liteモデル用のリクエストペイロードを構築
        # 会話履歴を含める
        bedrock_messages = []
        for msg in messages:
            if msg["role"] == "user":
                bedrock_messages.append({
                    "role": "user",
                    "content": [{"text": msg["content"]}]
                })
            elif msg["role"] == "assistant":
                bedrock_messages.append({
                    "role": "assistant", 
                    "content": [{"text": msg["content"]}]
                })
        
        # invoke_model用のリクエストペイロード
        request_payload = {
            "messages": bedrock_messages,
        }
        
        print("Calling Bedrock invoke_model API with payload:", json.dumps(request_payload))
        
        result = client.generate([
            {"prompt": message} #ここをrequest_payload
        ])
        
        # レスポンスを解析
        print("Bedrock response:", result)

        
        # アシスタントの応答を取得
        assistant_response = result['generated_text']
        
        # アシスタントの応答を会話履歴に追加
        messages.append({
            "role": "assistant",
            "content": assistant_response
        })
        
        # 成功レスポンスの返却
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": True,
                "response": assistant_response,
                "conversationHistory": messages
            })
        }
        
    except Exception as error:
        print("Error:", str(error))
        
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": False,
                "error": str(error)
            })
        }