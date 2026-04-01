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

    // ────────────── CD (로컬 실행) ──────────────

    stage('deploy') {
      steps {
        withCredentials([usernamePassword(
          credentialsId: 'harbor-cred',
          usernameVariable: 'HARBOR_USER',
          passwordVariable: 'HARBOR_PASS'
        )]) {
          sh """
            echo "Deploying locally (no SSH)..."

            echo \$HARBOR_PASS | docker login ${REGISTRY} \
              -u \$HARBOR_USER --password-stdin

            docker pull ${BACKEND_IMG}:${IMAGE_TAG}
            docker pull ${FRONTEND_IMG}:${IMAGE_TAG}

            cat > docker-compose.deploy.yml << "EOF"
version: "3.8"
services:
  backend:
    image: ${BACKEND_IMG}:${IMAGE_TAG}
    container_name: rag-backend
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
    restart: unless-stopped

  frontend:
    image: ${FRONTEND_IMG}:${IMAGE_TAG}
    container_name: rag-frontend
    ports:
      - "8501:8501"
    depends_on:
      - backend
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
        sh '''
          echo "Waiting for services..."
          sleep 15

          curl -f http://localhost:8000/health
          echo "Backend OK"

          curl -f -o /dev/null -w "%{http_code}" http://localhost:8501 | grep -q 200
          echo "Frontend OK"
        '''
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
        docker pull ${BACKEND_IMG}:latest || true
        docker pull ${FRONTEND_IMG}:latest || true

        docker compose -f docker-compose.deploy.yml up -d || true
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