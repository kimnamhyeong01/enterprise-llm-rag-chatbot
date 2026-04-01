pipeline {
  agent any

  environment {
    REGISTRY     = 'amdp-registry.skala-ai.com'
    PROJECT      = 'skala26a-ai2'
    BACKEND_IMG  = "${REGISTRY}/${PROJECT}/my-backend"
    FRONTEND_IMG = "${REGISTRY}/${PROJECT}/my-frontend"
    IMAGE_TAG    = "${BUILD_NUMBER}"
    DEPLOY_DIR   = '/opt/rag-chatbot'
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
          sshagent(credentials: ['deploy-server-ssh']) {
            sh """
              ssh -o StrictHostKeyChecking=no deploy@\${DEPLOY_HOST} '
                set -e

                mkdir -p ${DEPLOY_DIR}
                cd ${DEPLOY_DIR}

                cat > docker-compose.deploy.yml << "EOF"
version: "3.8"
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
    environment:
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - OPENAI_MODEL=\${OPENAI_MODEL:-gpt-4o-mini}
      - OPENAI_EMBEDDING_MODEL=\${OPENAI_EMBEDDING_MODEL:-text-embedding-3-small}
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

                echo "\$HARBOR_PASS" | docker login ${REGISTRY} \
                  -u "\$HARBOR_USER" --password-stdin

                docker pull ${BACKEND_IMG}:${IMAGE_TAG}
                docker pull ${FRONTEND_IMG}:${IMAGE_TAG}

                docker compose -f docker-compose.deploy.yml up -d --remove-orphans
              '
            """
          }
        }
      }
    }

    stage('verify') {
      steps {
        sh 'sleep 20'

        sshagent(credentials: ['deploy-server-ssh']) {
          sh """
            ssh -o StrictHostKeyChecking=no deploy@\${DEPLOY_HOST} '
              curl --retry 5 --retry-delay 5 -f http://localhost:8000/health
              echo "Backend OK"

              curl --retry 3 --retry-delay 5 -f -o /dev/null -w "%{http_code}" http://localhost:8501 | grep -q 200
              echo "Frontend OK"
            '
          """
        }
      }
    }
  }

  post {
    success {
      echo "CD 완료 - 빌드 #${BUILD_NUMBER} 배포 성공"
    }

    failure {
      echo "배포 실패 - 롤백 시도"

      sshagent(credentials: ['deploy-server-ssh']) {
        sh """
          ssh -o StrictHostKeyChecking=no deploy@\${DEPLOY_HOST} '
            cd ${DEPLOY_DIR}

            PREV=\$((${BUILD_NUMBER} - 1))
            if [ "\$PREV" -gt 0 ]; then
              docker pull ${BACKEND_IMG}:latest
              docker pull ${FRONTEND_IMG}:latest

              docker compose -f docker-compose.deploy.yml up -d
              echo "롤백 완료 (latest)"
            fi
          '
        """
      }
    }

    always {
      sh """
        docker rmi my-backend:${IMAGE_TAG} || true
        docker rmi my-frontend:${IMAGE_TAG} || true
      """
    }
  }
}