@Library('xmos_jenkins_shared_library@v0.28.0')

def runningOn(machine) {
  println "Stage running on:"
  println machine
}

getApproval()
pipeline {
  agent none

  options {
    skipDefaultCheckout()
    timestamps()
    buildDiscarder(xmosDiscardBuildSettings(onlyArtifacts=false))
  } // options

  stages {
    stage ('Build & Test') {
      agent {
        label 'linux&&x86_64'
      }
      stages {
        stage ('Build') {
          steps {
            runningOn(env.NODE_NAME)
            sh 'git clone -b develop git@github.com:xmos/xcommon_cmake'
            sh 'git -C xcommon_cmake rev-parse HEAD'
            dir("lib_audio_dsp") {
              checkout scm
              // build everything
              withTools(params.TOOLS_VERSION) {
                withEnv(["XMOS_CMAKE_PATH=${WORKSPACE}/xcommon_cmake"]) {
                  script {
                    [
                     "test/biquad"
                    ].each {
                      sh "cmake -S ${it} -B ${it}/build"
                      sh "xmake -C ${it}/build -j"
                    }
                  }
                }
              }
            }
          }
        } // Build

        stage ('Create Python enviroment') {
          steps {
            dir("lib_audio_dsp") {
              createVenv("requirements.txt")
            }
          }
        } // Create Python enviroment

        stage ('Test') {
          steps {
            dir("lib_audio_dsp") {
              withVenv {
                withTools(params.TOOLS_VERSION) {
                  dir("test/biquad") {
                    // running separately because it's faster for parallel tests
                    sh "pytest -s test_biquad_python.py"
                    sh "pytest -n auto test_biquad_c.py"
                  }
                }
              }
            }
          }
        } // Test

      } // stages
      post {
        cleanup {
          xcoreCleanSandbox()
        }
      }
    } // Build & Test
  } // stages
} // pipeline
