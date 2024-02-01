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

  environment {
    XMOSDOC_VERSION = "v5.1.0"
  } // environment

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

            }
            createVenv("lib_audio_dsp/requirements.txt")

            dir("lib_audio_dsp") {
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

        stage('test and docs') {
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
                withVenv {
                  withTools(params.TOOLS_VERSION) {
                    dir('lib_audio_dsp') {
                      sh "python doc/programming_guide/gen/autogen.py"
                      sh """docker run -u "\$(id -u):\$(id -g)" \
                            --rm \
                            -v ${WORKSPACE}/lib_audio_dsp:/build \
                            --entrypoint /build/doc/build_docs.sh \
                            ghcr.io/xmos/xmosdoc:$XMOSDOC_VERSION -v"""
                      archiveArtifacts artifacts: "doc/_out/pdf/*.pdf"
                      archiveArtifacts artifacts: "doc/_out/html/**/*"
                      archiveArtifacts artifacts: "doc/_out/linkcheck/**/*"
                    }
                  }
                }
              }
            } // Docs
          } // parallel
        } // test and docs

      } // stages
      post {
        cleanup {
          xcoreCleanSandbox()
        }
      }
    } // Build & Test
  } // stages
} // pipeline
