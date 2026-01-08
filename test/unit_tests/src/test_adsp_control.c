// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

/// Unit tests for adsp_control.c

#include <unity.h>

#include <stages/adsp_control.h>
#include <stages/dummy.h>
#include <adsp_generated_dummy.h>
#include <adsp_instance_id_dummy.h>
#include <cmds.h>
#include <stdio.h>
#include <swlock.h>

void setUp(){}
void tearDown(){}

/// Test that basic read and write works.
void test_basic_control(void) {

    adsp_pipeline_t* p = adsp_dummy_pipeline_init();
    module_instance_t* dummy_inst = &p->modules[dummy_stage_index];
    adsp_controller_t ctrl;
    adsp_controller_init(&ctrl, p);

    int test_val = 5;
    adsp_stage_control_cmd_t cmd = {
        .instance_id = dummy_stage_index,
        .cmd_id = CMD_DUMMY_DUMMY_FIELD,
        .payload_len = sizeof(int),
        .payload = &test_val
    };

    // First write succeeds as stage isn't busy
    adsp_control_status_t status =  adsp_write_module_config(&ctrl, &cmd);
    TEST_ASSERT_EQUAL(ADSP_CONTROL_SUCCESS, status);

    // Now the stage is busy
    status =  adsp_write_module_config(&ctrl, &cmd);
    TEST_ASSERT_EQUAL(ADSP_CONTROL_BUSY, status);
    // Now the stage is busy
    status =  adsp_read_module_config(&ctrl, &cmd);
    TEST_ASSERT_EQUAL(ADSP_CONTROL_BUSY, status);

    // Stage processes control
    dummy_control(dummy_inst->state, &dummy_inst->control);

    *(int*)cmd.payload = 0; // clear the payload, ready for read.

    // request a read, stage must process it so it is busy
    status =  adsp_read_module_config(&ctrl, &cmd);
    TEST_ASSERT_EQUAL(ADSP_CONTROL_BUSY, status);

    // The stage is busy
    status =  adsp_write_module_config(&ctrl, &cmd);
    TEST_ASSERT_EQUAL(ADSP_CONTROL_BUSY, status);

    // Stage processes control
    dummy_control(dummy_inst->state, &dummy_inst->control);


    // re-request the read, stage must process it so it is busy
    status =  adsp_read_module_config(&ctrl, &cmd);
    TEST_ASSERT_EQUAL(ADSP_CONTROL_SUCCESS, status);
    TEST_ASSERT_EQUAL(test_val, *(int*)cmd.payload);
}

/// Test that when there are two controllers, only the controller who made
/// the request will get a response.
void test_adsp_control_2_controllers(void) {
    adsp_pipeline_t* p = adsp_dummy_pipeline_init();
    module_instance_t* dummy_inst = &p->modules[dummy_stage_index];
    adsp_controller_t ctrl_a;
    adsp_controller_init(&ctrl_a, p);
    adsp_controller_t ctrl_b;
    adsp_controller_init(&ctrl_b, p);

    int test_val = 5;
    adsp_stage_control_cmd_t cmd = {
        .instance_id = dummy_stage_index,
        .cmd_id = CMD_DUMMY_DUMMY_FIELD,
        .payload_len = sizeof(int),
        .payload = &test_val
    };

    // First write succeeds as stage isn't busy
    adsp_control_status_t status =  adsp_write_module_config(&ctrl_a, &cmd);
    TEST_ASSERT_EQUAL(ADSP_CONTROL_SUCCESS, status);
    // stage is now busy for both controllers.
    status =  adsp_write_module_config(&ctrl_a, &cmd);
    TEST_ASSERT_EQUAL(ADSP_CONTROL_BUSY, status);
    status =  adsp_write_module_config(&ctrl_b, &cmd);
    TEST_ASSERT_EQUAL(ADSP_CONTROL_BUSY, status);

    // Stage processes control
    dummy_control(dummy_inst->state, &dummy_inst->control);

    *(int*)cmd.payload = 0; // clear the payload, ready for read.

    // request a read, stage must process it so it is busy
    status =  adsp_read_module_config(&ctrl_a, &cmd);
    TEST_ASSERT_EQUAL(ADSP_CONTROL_BUSY, status);

    // Stage processes control
    dummy_control(dummy_inst->state, &dummy_inst->control);


    // Even though a read was processed, b still gets busy
    // because it didn't ask for it.
    status =  adsp_read_module_config(&ctrl_b, &cmd);
    TEST_ASSERT_EQUAL(ADSP_CONTROL_BUSY, status);
    // but a gets the result.
    status =  adsp_read_module_config(&ctrl_a, &cmd);
    TEST_ASSERT_EQUAL(ADSP_CONTROL_SUCCESS, status);
    TEST_ASSERT_EQUAL(test_val, *(int*)cmd.payload);
}


/// Ensure that busy is returned when the lock is taken for both read
/// and write.
void test_lock_contention(void) {

    adsp_pipeline_t* p = adsp_dummy_pipeline_init();
    module_instance_t* dummy_inst = &p->modules[dummy_stage_index];
    adsp_controller_t ctrl;
    adsp_controller_init(&ctrl, p);

    int test_val = 5;
    adsp_stage_control_cmd_t cmd = {
        .instance_id = dummy_stage_index,
        .cmd_id = CMD_DUMMY_DUMMY_FIELD,
        .payload_len = sizeof(int),
        .payload = &test_val
    };

    // take the lock to emulate contention.
    swlock_acquire(&dummy_inst->control.lock);

    // Get busy even though the stage isn't doing anything because the lock
    // is taken.
    adsp_control_status_t status =  adsp_write_module_config(&ctrl, &cmd);
    TEST_ASSERT_EQUAL(ADSP_CONTROL_BUSY, status);

    swlock_release(&dummy_inst->control.lock);

    status =  adsp_write_module_config(&ctrl, &cmd);
    TEST_ASSERT_EQUAL(ADSP_CONTROL_SUCCESS, status);

    // Stage processes control
    dummy_control(dummy_inst->state, &dummy_inst->control);
}

