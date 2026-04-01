pipeline {
  agent any

  environment {
    REGISTRY     = 'amdp-registry.skala-ai.com'
    PROJECT      = 'skala26a-ai2'
    BACKEND_IMG  = "${REGISTRY}/${PROJECT}/my-backend"
    FRONTEND_IMG = "${REGISTRY}/${PROJECT}/my-frontend"
    IMAGE_TAG    = "${BUILD_NUMBER}"
  }

  stages {

    // ────────────── CI ──────────────

    stage('build-backend') {
      steps {
        sh "docker build -t my-backend:${IMAGE_TAG} ./backend"
      }
    }

    stage('build-frontend') {
      steps {
        sh "docker build -t my-frontend:${IMAGE_TAG} ./frontend"
      }
    }

    stage('test') {
      steps {
        sh '''
          echo "Running backend smoke test..."
          docker run --rm \
            -e OPENAI_API_KEY=dummy \
            my-backend:${IMAGE_TAG} \
            python -c "from main import app; print('Import OK')"
        '''
      }
    }

    stage('push') {
      steps {
        withCredentials([usernamePassword(
          credentialsId: 'harbor-cred',
          usernameVariable: 'HARBOR_USER',
          passwordVariable: 'HARBOR_PASS'
        )]) {
          sh """
            echo \$HARBOR_PASS | docker login ${REGISTRY} \
              -u \$HARBOR_USER --password-stdin

            docker tag my-backend:${IMAGE_TAG}  ${BACKEND_IMG}:${IMAGE_TAG}
            docker tag my-backend:${IMAGE_TAG}  ${BACKEND_IMG}:latest
            docker tag my-frontend:${IMAGE_TAG} ${FRONTEND_IMG}:${IMAGE_TAG}
            docker tag my-frontend:${IMAGE_TAG} ${FRONTEND_IMG}:latest

            docker push ${BACKEND_IMG}:${IMAGE_TAG}
            docker push ${BACKEND_IMG}:latest
            docker push ${FRONTEND_IMG}:${IMAGE_TAG}
            docker push ${FRONTEND_IMG}:latest
          """
        }
      }
    }

    // ────────────── CD ──────────────

    stage('deploy') {
      steps {
        withCredentials([usernamePassword(
          credentialsId: 'harbor-cred',
          usernameVariable: 'HARBOR_USER',
          passwordVariable: 'HARBOR_PASS'
        )]) {
          sh """
            echo "Deploying..."

            echo \$HARBOR_PASS | docker login ${REGISTRY} \
              -u \$HARBOR_USER --password-stdin

            docker pull ${BACKEND_IMG}:${IMAGE_TAG}
            docker pull ${FRONTEND_IMG}:${IMAGE_TAG}

            # ✅ 기존 서비스만 제거 (Jenkins 유지)
            docker rm -f rag-backend rag-frontend || true

            # compose 파일 생성
            cat > docker-compose.deploy.yml << "EOF"
services:
  backend:
    image: ${BACKEND_IMG}:${IMAGE_TAG}
    container_name: rag-backend
    ports:
      - "8000:8000"
    volumes:
      - ./knowledge:/app/knowledge
      - ./docs:/app/docs
      - ./chroma_db:/app/chroma_db
    env_file:
      - .env
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    image: ${FRONTEND_IMG}:${IMAGE_TAG}
    container_name: rag-frontend
    ports:
      - "8501:8501"
    depends_on:
      backend:
        condition: service_healthy
    environment:
      - BACKEND_URL=http://backend:8000
    restart: unless-stopped
EOF

            docker compose -f docker-compose.deploy.yml up -d --remove-orphans
          """
        }
      }
    }

    stage('verify') {
      steps {
        sh 'sleep 20'

        sh """
          curl --retry 5 --retry-delay 5 -f http://localhost:8000/health
          echo "Backend OK"

          curl --retry 3 --retry-delay 5 -f -o /dev/null -w "%{http_code}" http://localhost:8501 | grep -q 200
          echo "Frontend OK"
        """
      }
    }
  }

  post {
    success {
      echo "CD 완료 - 빌드 #${BUILD_NUMBER} 배포 성공"
    }

    failure {
      echo "배포 실패 - 롤백 시도"

      sh """
        docker pull ${BACKEND_IMG}:latest
        docker pull ${FRONTEND_IMG}:latest

        docker rm -f rag-backend rag-frontend || true

        docker compose -f docker-compose.deploy.yml up -d
      """
    }

    always {
      sh """
        docker rmi my-backend:${IMAGE_TAG} || true
        docker rmi my-frontend:${IMAGE_TAG} || true
      """
    }
  }
}