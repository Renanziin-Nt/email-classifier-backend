import pickle
from pathlib import Path
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MLModel:
    def __init__(self, model_path: str = "models/classifier_model.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        self.training_data = self._load_training_data()
        
    def _load_training_data(self):
        return {
            "Produtivo": [
                "Preciso de ajuda urgente com erro no sistema de pagamento",
                "Qual o status atual da minha solicitação de reembolso #12345?",
                "Como proceder para atualizar meus dados cadastrais na plataforma?",
                "Estou com problemas críticos para acessar minha conta corporativa",
                "Solicito suporte técnico imediato para falha no processo de login",
                "Gostaria de saber os prazos exatos de entrega do pedido #67890",
                "Preciso falar urgentemente com o departamento financeiro sobre cobrança indevida",
                "Como faço para resetar minha senha de acesso ao sistema?",
                "Relato um bug crítico na funcionalidade de upload de arquivos grandes",
                "Solicito informações detalhadas sobre o andamento do meu caso #11223",
                "Problema emergencial no sistema requer atenção imediata da equipe",
                "Não consigo acessar meus relatórios financeiros mensais",
                "Erro 500 persistente no processamento de transações cartão de crédito",
                "Solicitação de prioridade máxima para pedido importante do cliente XYZ",
                "Dúvida técnica sobre integração da API de pagamentos com nosso ERP",
                "Problema de performance na dashboard administrativa",
                "Relatório de inconsistência nos dados de faturamento",
                "Solicitação de novo recurso para exportação de relatórios",
                "Problema de conectividade com servidor de produção",
                "Dúvida sobre configuração de webhook para notificações"
            ],
            "Improdutivo": [
                "Agradeço profundamente pela excelente assistência técnica recebida",
                "Desejo um feliz natal e próspero ano novo a toda equipe maravilhosa",
                "Parabéns pelo ótimo trabalho desenvolvido pela equipe de suporte",
                "Apenas passando para desejar um bom final de semana prolongado",
                "Obrigado pelo atendimento de qualidade excepcional prestado",
                "Feliz páscoa para toda a família corporativa maravilhosa",
                "Cumprimentos e saudações cordiais a todos os colaboradores",
                "Agradeço sinceramente a atenção dedicada de todos os envolvidos",
                "Desejo muito sucesso nos projetos futuros da empresa",
                "Mensagem de agradecimento pelo apoio incansável recebido",
                "Feliz ano novo cheio de conquistas e realizações profissionais",
                "Parabéns pelo aniversário de 5 anos da empresa brilhante",
                "Agradecimento pelo suporte excepcional prestado durante o projeto",
                "Desejo um ótimo feriado prolongado para todos os colaboradores",
                "Mensagem de carinho e apreciação pelo trabalho dedicado",
                "Cumprimentos festivos pela época natalina que se aproxima",
                "Agradecimento pela paciência e profissionalismo demonstrados",
                "Desejo um final de semana relaxante e revigorante",
                "Mensagem de reconhecimento pelo excelente atendimento ao cliente",
                "Cumprimentos cordiais e votos de prosperidade para todos"
            ]
        }
    
    def train(self):
        """Treina o modelo de Machine Learning"""
        try:
            texts, labels = self._prepare_training_data()
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
                ('clf', LogisticRegression(random_state=42, max_iter=1000))
            ])
            
            self.model.fit(X_train, y_train)
            
            # Validação
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            logger.info(f"Modelo treinado com acurácia: {accuracy:.2f}")
            return accuracy
            
        except Exception as e:
            logger.error(f"Erro no treinamento: {e}")
            raise
    
    def predict(self, text: str):
        """Faz predição com o modelo treinado"""
        if not self.is_trained:
            self.train()
        
        prediction = self.model.predict([text])[0]
        probabilities = self.model.predict_proba([text])[0]
        confidence = max(probabilities)
        
        return prediction, float(confidence)
    
    def save(self):
        """Salva o modelo treinado"""
        try:
            self.model_path.parent.mkdir(exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'training_date': datetime.now(),
                    'accuracy': self._get_accuracy()
                }, f)
            logger.info("Modelo salvo com sucesso")
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {e}")
    
    def load(self):
        """Carrega modelo salvo"""
        try:
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.model = saved_data['model']
                    self.is_trained = True
                logger.info("Modelo carregado do cache")
                return True
            return False
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            return False
    
    def _prepare_training_data(self):
        """Prepara dados para treinamento"""
        texts = []
        labels = []
        
        for label, examples in self.training_data.items():
            texts.extend(examples)
            labels.extend([label] * len(examples))
        
        return texts, labels
    
    def _get_accuracy(self):
        """Calcula acurácia do modelo"""
        if not self.is_trained:
            return 0.0
        
        texts, labels = self._prepare_training_data()
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        y_pred = self.model.predict(X_test)
        return accuracy_score(y_test, y_pred)