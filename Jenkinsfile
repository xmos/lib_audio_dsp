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
    XMOSDOC_VERSION = "v5.1.1"
  } // environment

  options {
    skipDefaultCheckout()
    timestamps()
    buildDiscarder(xmosDiscardBuildSettings(onlyArtifacts=false))
  } // options

  stages {
    stage('CI') {
      parallel {
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
                    withTools(params.TOOLS_VERSION) {
                      sh "pip install -r requirements.txt"
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

            stage('test') {
              steps {
                dir("lib_audio_dsp") {
                  withVenv {
                    withTools(params.TOOLS_VERSION) {
                      dir("test/biquad") {
                        runPytest("test_biquad_python.py --dist worksteal")
                        runPytest("test_biquad_c.py --dist worksteal")
                      }
                      dir("test/cascaded_biquads") {
                        runPytest("test_cascaded_biquads_python.py --dist worksteal")
                        runPytest("test_cascaded_biquads_c.py --dist worksteal")
                      }
                      dir("test/drc") {
                        runPytest("test_drc_python.py --dist worksteal")
                        runPytest("test_drc_c.py --dist worksteal")
                      }
                      dir("test/utils") {
                        runPytest("--dist worksteal")
                      }
                    }
                  }
                }
              }
            } // test
          }

          post {
            cleanup {
              xcoreCleanSandbox()
            }
          }
        } // Build and test

        stage('docs') {

          agent {
            label 'linux&&x86_64'
          }
          steps {
            checkout scm
            sh """docker run -u "\$(id -u):\$(id -g)" \
                  --rm \
                  -v ${WORKSPACE}:/build \
                  --entrypoint /build/doc/build_docs.sh \
                  ghcr.io/xmos/xmosdoc:$XMOSDOC_VERSION -v"""
            archiveArtifacts artifacts: "doc/_out/pdf/*.pdf"
            archiveArtifacts artifacts: "doc/_out/html/**/*"
            archiveArtifacts artifacts: "doc/_out/linkcheck/**/*"
          }
          post {
            cleanup {
              xcoreCleanSandbox()
            }
          }
        } // docs

        stage ('Hardware Test') {
          agent {
            label 'xcore.ai && uhubctl'
          }

          steps {
            runningOn(env.NODE_NAME)
            sh 'git clone -b develop git@github.com:xmos/xcommon_cmake'
            sh 'git -C xcommon_cmake rev-parse HEAD'
            sh 'git clone https://github0.xmos.com/xmos-int/xtagctl.git'
            dir("lib_audio_dsp") {
              checkout scm
            }
            createVenv("lib_audio_dsp/requirements.txt")

            dir("lib_audio_dsp") {
              withVenv {
                withTools(params.TOOLS_VERSION) {
                  sh "pip install -r requirements.txt"
                  sh "pip install -e ${WORKSPACE}/xtagctl"
                  withEnv(["XMOS_CMAKE_PATH=${WORKSPACE}/xcommon_cmake"]) {
                    withXTAG(["XCORE-AI-EXPLORER"]) { adapterIDs ->
                      dir("test/pipeline") {
                        sh "python -m pytest --junitxml=pytest_result.xml -rA -v --durations=0 -o junit_logging=all --log-cli-level=INFO --adapter-id " + adapterIDs[0]
                      }
                    }
                  }
                }
              }
            }
          }

          post {
            cleanup {
              xcoreCleanSandbox()
            }
            always {
              dir("${WORKSPACE}/lib_audio_dsp/test/pipeline") {
                junit "pytest_result.xml"
              }
            }
          }
        } // Hardware test
      } // parallel
    } // CI
  } // stages
} // pipeline
