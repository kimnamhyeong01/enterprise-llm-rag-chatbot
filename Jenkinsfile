pipeline {
  agent any

  stages {

    stage('build-backend') {
      steps {
        sh 'docker build -t my-backend ./backend'
      }
    }

    stage('build-frontend') {
      steps {
        sh 'docker build -t my-frontend ./frontend'
      }
    }

    stage('test') {
      steps {
        sh '''
        echo "Running tests..."
        # 여기에 pytest 등 실제 테스트 넣어도 됨
        '''
      }
    }

    stage('tag') {
      steps {
        sh '''
        docker tag my-backend amdp-registry.skala-ai.com/skala26a-ai2/my-backend:latest
        docker tag my-frontend amdp-registry.skala-ai.com/skala26a-ai2/my-frontend:latest
        '''
      }
    }

    stage('deploy') {
      steps {
        withCredentials([usernamePassword(
          credentialsId: 'harbor-cred',
          usernameVariable: 'HARBOR_USER',
          passwordVariable: 'HARBOR_PASS'
        )]) {
          sh '''
          echo $HARBOR_PASS | docker login amdp-registry.skala-ai.com \
            -u $HARBOR_USER --password-stdin

          docker push amdp-registry.skala-ai.com/skala26a-ai2/my-backend:latest
          docker push amdp-registry.skala-ai.com/skala26a-ai2/my-frontend:latest
          '''
        }
      }
    }

  }
}