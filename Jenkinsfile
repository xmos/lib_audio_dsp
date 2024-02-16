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
    string(
      name: 'XCOMMON_CMAKE_VERSION',
      defaultValue: 'v0.2.0',
      description: 'The xcommon cmake version'
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
                sh "git clone -b ${params.XCOMMON_CMAKE_VERSION} git@github.com:xmos/xcommon_cmake"
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
                        println 'runPytest("test_biquad_python.py --dist worksteal")'
                        println 'runPytest("test_biquad_c.py --dist worksteal")'
                      }
                      dir("test/cascaded_biquads") {
                        println '"test_cascaded_biquads_python.py --dist worksteal")'
                        println 'runPytest("test_cascaded_biquads_c.py --dist worksteal")'
                      }
                      dir("test/drc") {
                        println 'runPytest("test_drc_python.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                      }
                      dir("test/utils") {
                        println 'runPytest("--dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                        println 'runPytest("test_drc_c.py --dist worksteal")'
                      }
                      dir("python") {
                        sh "pyright audio_dsp --skipunannotated --level warning"
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
            println "nuilding docs"
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

            dir("lib_audio_dsp") {
              println 'hardware testing'
            }
          }

          post {
            cleanup {
              xcoreCleanSandbox()
            }
          }
        } // Hardware test        
      } // parallel
    } // CI
  } // stages
} // pipeline
