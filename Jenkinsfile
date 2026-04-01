pipeline {
  agent any

  environment {
    REGISTRY     = 'amdp-registry.skala-ai.com'
    PROJECT      = 'skala26a-ai2'
    BACKEND_IMG  = "${REGISTRY}/${PROJECT}/sk035-backend"
    FRONTEND_IMG = "${REGISTRY}/${PROJECT}/sk035-frontend"
    IMAGE_TAG    = "${BUILD_NUMBER}"
  }

  stages {

    // ────────────── CI ──────────────

    stage('build-backend') {
      steps {
        sh "docker build -t sk035-backend:${IMAGE_TAG} ./backend"
      }
    }

    stage('build-frontend') {
      steps {
        sh "docker build -t sk035-frontend:${IMAGE_TAG} ./frontend"
      }
    }

    stage('test') {
      steps {
        sh '''
          echo "Running backend smoke test..."
          docker run --rm \
            -e OPENAI_API_KEY=dummy \
            sk035-backend:${IMAGE_TAG} \
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

            docker tag sk035-backend:${IMAGE_TAG}  ${BACKEND_IMG}:${IMAGE_TAG}
            docker tag sk035-backend:${IMAGE_TAG}  ${BACKEND_IMG}:latest
            docker tag sk035-frontend:${IMAGE_TAG} ${FRONTEND_IMG}:${IMAGE_TAG}
            docker tag sk035-frontend:${IMAGE_TAG} ${FRONTEND_IMG}:latest

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
        withCredentials([
          file(credentialsId: 'kubeconfig', variable: 'KUBECONFIG')
        ]) {
          sh """
            set -e
            export KUBECONFIG=\$KUBECONFIG

            echo "Deploying to Kubernetes..."

            # 이미지 태그를 현재 빌드 번호로 업데이트
            kubectl set image deployment/rag-app \
              backend=${BACKEND_IMG}:${IMAGE_TAG} \
              frontend=${FRONTEND_IMG}:${IMAGE_TAG}

            # 롤아웃 완료까지 대기 (최대 3분)
            kubectl rollout status deployment/rag-app --timeout=180s
          """
        }
      }
    }

    stage('verify') {
      steps {
        withCredentials([
          file(credentialsId: 'kubeconfig', variable: 'KUBECONFIG')
        ]) {
          sh """
            export KUBECONFIG=\$KUBECONFIG

            kubectl exec deploy/rag-app -c backend -- curl -sf http://localhost:8000/health
            echo "Backend OK"

            kubectl exec deploy/rag-app -c frontend -- python -c "import urllib.request; r = urllib.request.urlopen('http://localhost:8501'); exit(0 if r.status == 200 else 1)"
            echo "Frontend OK"
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
      echo "배포 실패 - 이전 버전으로 롤백"

      withCredentials([file(credentialsId: 'kubeconfig', variable: 'KUBECONFIG')]) {
        sh """
          export KUBECONFIG=\$KUBECONFIG
          kubectl rollout undo deployment/rag-app
        """
      }
    }

    always {
      sh """
        docker rmi sk035-backend:${IMAGE_TAG} || true
        docker rmi sk035-frontend:${IMAGE_TAG} || true
      """
    }
  }
}