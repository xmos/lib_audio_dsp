@Library('xmos_jenkins_shared_library@v0.28.0')

def runningOn(machine) {
  println "Stage running on:"
  println machine
}

getApproval()
pipeline {
  agent none

  parameters {
    string(
      name: 'TOOLS_VERSION',
      defaultValue: '15.2.1',
      description: 'The XTC tools version'
    )
  } // parameters

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
              // try building a simple app without venv to check
              // build that doesn't use design tools won't
              // need python
              withTools(params.TOOLS_VERSION) {
                withEnv(["XMOS_CMAKE_PATH=${WORKSPACE}/xcommon_cmake"]) {
                  dir("test/biquad") {
                    sh "cmake -B build"
                    sh "cmake --build build"
                  }
                }
              }

              createVenv("requirements.txt")
              // build everything
              withVenv {
                sh "pip install -r requirements.txt"
                withTools(params.TOOLS_VERSION) {
                  withEnv(["XMOS_CMAKE_PATH=${WORKSPACE}/xcommon_cmake"]) {
                    script {
                      [
                      "test/biquad",
                      "test/cascaded_biquads",
                      "test/drc"
                      ].each {
                        sh "cmake -S ${it} -B ${it}/build"
                        sh "xmake -C ${it}/build -j"
                      }
                    }
                  }
                }
              }
            }
          }
        } // Build

        parallel {
          stage ('Test') {
            steps {
              dir("lib_audio_dsp") {
                withVenv {
                  withTools(params.TOOLS_VERSION) {
                    dir("test/biquad") {
                      runPytest("--dist worksteal")
                    }
                    dir("test/cascaded_biquads") {
                      runPytest("--dist worksteal")
                    }
                    dir("test/drc") {
                      runPytest("--dist worksteal")
                    }
                    dir("test/utils") {
                      runPytest("--dist worksteal")
                    }
                  }
                }
              }
            }
          } // Test

          stage ('Docs') {
            steps {
              sh 'git clone git@github.com:xmos/xmosdoc'

              withVenv {
                withTools(params.TOOLS_VERSION) {
                  sh 'pip install -e xmosdoc'
                  dir('lib_audio_dsp') {
                    sh 'xmosdoc -dvvv'
                  }
                }
              }
            }
          } // Docs
        } // parallel

      } // stages
      post {
        cleanup {
          xcoreCleanSandbox()
        }
      }
    } // Build & Test
  } // stages
} // pipeline
