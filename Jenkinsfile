pipeline {
  agent any

  stages {

    stage('build') {
      steps {
        sh 'docker build -t my-backend ./backend'
        sh 'docker build -t my-frontend ./frontend'
      }
    }

    stage('test') {
      steps {
        sh '''
        echo "Running tests..."
        # 실제 테스트 넣고 싶으면 아래처럼
        # pytest ./backend
        # npm test (frontend일 경우)
        '''
      }
    }

    stage('deploy') {
      steps {
        sh '''
        docker tag my-backend amdp-registry.skala-ai.com/skala26a-ai2/my-backend:latest
        docker tag my-frontend amdp-registry.skala-ai.com/skala26a-ai2/my-frontend:latest

        docker login amdp-registry.skala-ai.com \
        -u 'robot$skala26a-ai2' \
        -p Va9M8WvbaoPa4oxpqFHMH4TH0h02GbTH

        docker push amdp-registry.skala-ai.com/skala26a-ai2/my-backend:latest
        docker push amdp-registry.skala-ai.com/skala26a-ai2/my-frontend:latest
        '''
      }
    }

  }
}